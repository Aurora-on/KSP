"""Calibrate an S3 world model and adapt its actor on separate target-domain interactions."""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import json
from itertools import combinations
from pathlib import Path
import time

import numpy as np
import torch
from mun_lander.evaluation import generate_initial_states

from . import nominal
from .dataset import file_hash, load_dataset, save_dataset
from .deterministic_fit import EpisodeWindows
from .fit import code_version
from .imagination import Actor, Critic, update_actor_critic
from .policy import Policy, policy_checkpoint
from .replay import Replay
from .train import _counts, _near, _sample_starts, _write_json, load_training_world_model
from .transfer import target_config


SEEDS = (31, 32, 33)
CHECKPOINT_SHA = "a4451b97f733712c4d7832572d787ff24b816d214bb9f55fc03cdfc0837eae65"
PREFILL_SHA = "b26f50ac573ea3f7fa1113cc7137bb4868cf3757caf4e53a7e2fb33ac269021c"
MANIFEST_SHA = {
    "validation": "2bf7a0898b9ed4f38dd46f996765f675a97fe703f0c9dcc0fee83727d0e51321",
    "test": "c12a88bc1047314fb482bbeec67ba5395e608cdd8be444e7f212f23e8a483f35",
}


def verify_inputs(checkpoint, source_meta, validation_manifest, test_manifest):
    """Verify identities after load_dataset and load_manifest have validated contents."""
    if file_hash(Path(checkpoint)) != CHECKPOINT_SHA or source_meta["replay_sha256"] != PREFILL_SHA:
        raise ValueError("adaptation requires the frozen seed-33 round-22 model and S3 source prefill")
    for name, manifest in [("validation", validation_manifest), ("test", test_manifest)]:
        if manifest["split"] != name or manifest["sha256"] != MANIFEST_SHA[name]:
            raise ValueError(f"unexpected {name} manifest")


def initialize_models(checkpoint):
    """Restore the three trained parameter sets, without restoring optimizer or RNG state."""
    saved = checkpoint if isinstance(checkpoint, dict) else torch.load(
        checkpoint, map_location="cpu", weights_only=True)
    config = saved["provenance"]["config"]
    actor = Actor(**saved["actor_spec"])
    actor.load_state_dict(saved["actor"])
    policy = Policy(actor.eval(), saved["interface_spec"])
    critic = Critic(config["critic_width"], config["value_scale"],
                    upper_bound=config.get("critic_upper_bound"),
                    yaw_invariant=config.get("yaw_invariant", False))
    critic.load_state_dict(saved["critic"])
    return policy, critic, load_training_world_model(saved, "cpu")


def check_seed_partition(*, target_train, target_heldout, source_prefill, validation, test):
    """Reject repeated reset seeds within or across the five independent groups."""
    groups = {}
    for name, seeds in (("target_train", target_train), ("target_heldout", target_heldout),
                        ("source_prefill", source_prefill), ("validation", validation), ("test", test)):
        values = list(seeds)
        groups[name] = set(values)
        if len(groups[name]) != len(values):
            raise ValueError(f"duplicate reset seeds within {name}")
    for left, right in combinations(groups, 2):
        overlap = groups[left] & groups[right]
        if overlap:
            raise ValueError(f"reset seed {min(overlap)} occurs in both {left} and {right}")


def _a1_dataset(domain, role, config, data_dir, policy):
    """Cache immutable frozen-policy rollouts; reject incompatible existing data."""
    cfg = target_config(domain)
    seed = config["collection_seeds"][domain][role]
    count = config[f"{role}_episodes"]
    expected = dict(domain=domain, role=role, config=cfg.to_dict(), seed=seed, episodes=count,
                    source_checkpoint_sha256=CHECKPOINT_SHA, noise_std=config["noise_std"])
    path = Path(data_dir) / f"adapt-{domain.lower()}-a1-{role}"
    if not path.exists():
        replay, meta = nominal.collect_policy(policy, seed=seed, episodes=count, noise_std=config["noise_std"], config=cfg)
        save_dataset(path, replay, dict(meta, adaptation=expected))
    replay, meta = load_dataset(path)
    if meta.get("adaptation") != expected:
        raise ValueError(f"A1 dataset provenance mismatch: {path}")
    return replay, meta


def world_optimizer(model, lr):
    """Use the S3 Adam parameter groups with fresh moments."""
    groups = [{"params": [*model.dynamics.parameters(), *model.event.parameters(),
                          *model.diagnostic_reward.parameters()]}]
    if model.kinematic:
        groups.append({"params": model.liftoff.parameters(), "lr": .001 * lr})
    return torch.optim.Adam(groups, lr=lr)


def fit_world_steps(model, optimizer, source, target, rng, *, updates, batch_size, source_fraction):
    """Fit the S3 eight-step losses using complete windows from each replay."""

    if type(updates) is not int or updates < 0 or batch_size < 2 or not 0 < source_fraction < 1:
        raise ValueError("invalid update count, mixed batch size, or source fraction")
    totals = {"dynamics_loss": 0., "event_loss": 0., "reward_loss": 0.}
    if not updates:
        return totals
    n_source, n_target = _counts(batch_size, source_fraction)
    parameters = [*model.dynamics.parameters(), *model.event.parameters()]
    if model.kinematic:
        parameters.extend(model.liftoff.parameters())
    model.train()
    for _ in range(updates):
        source_batch = source.sample(rng, n_source, 8)
        target_batch = target.sample(rng, n_target, 8)
        losses = model.losses(*(torch.cat(pair) for pair in zip(source_batch, target_batch, strict=True)))
        optimizer.zero_grad(set_to_none=True)
        sum(losses).backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        torch.nn.utils.clip_grad_norm_(model.diagnostic_reward.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        for key, loss in zip(totals, losses, strict=True):
            totals[key] += loss.item() / updates
    return totals


def _a1_reports(model, replay, meta, source_mean, label, out):
    """Evaluate original S1 gates; the cheap prefix is diagnostic only."""
    from .deterministic_report import deterministic_report

    result = {}
    for count in (20, 50):
        if len(replay.episodes) < count:
            raise ValueError(f"A1 requires {count} complete held-out episodes")
        report = deterministic_report(model, Replay(replay.episodes[:count]), model.spec,
                                      train_mean=source_mean, provenance=meta["episode_provenance"][:count])
        name = f"{label}-{count}.json"
        _write_json(out / name, report)
        gates = {key: bool(report["gates"][key]["pass"]) for key in ("W1", "W2", "W4", "W5")}
        result[str(count)] = {"report": name, "gates": gates, "all_four_pass": all(gates.values()),
                              "acceptance": report["acceptance"]["pass"]}
    return result


def run_a1(domain, config, *, checkpoint, prefill_path, data_dir, out, validation_path, test_path):
    """Evaluate the frozen source WM, then fit the predeclared main and conditional fallback."""
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["fit_seed"])
    checkpoint, out = Path(checkpoint), Path(out)
    source_replay, source_meta = load_dataset(prefill_path)
    validation = nominal.load_manifest(validation_path, "validation")
    test = nominal.load_manifest(test_path, "test")
    verify_inputs(checkpoint, source_meta, validation, test)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy, _, model = initialize_models(saved)
    training, train_meta = _a1_dataset(domain, "train", config, data_dir, policy)
    heldout, heldout_meta = _a1_dataset(domain, "heldout", config, data_dir, policy)
    check_seed_partition(target_train=train_meta["reset_seeds"], target_heldout=heldout_meta["reset_seeds"],
                         source_prefill=source_meta["reset_seeds"], validation=[s["seed"] for s in validation["starts"]],
                         test=[s["seed"] for s in test["starts"]])
    out.mkdir(parents=True, exist_ok=False)
    source = EpisodeWindows(source_replay, source_meta, model.spec, "cpu")
    target = EpisodeWindows(training, train_meta, model.spec, "cpu")
    summary = dict(schema_version=1, domain=domain, config=config, code=code_version(),
                   runner_sha256=file_hash(Path(__file__)),
                   source_checkpoint_sha256=CHECKPOINT_SHA, source_replay_sha256=source_meta["replay_sha256"],
                   train_reset_seeds=train_meta["reset_seeds"], heldout_reset_seeds=heldout_meta["reset_seeds"],
                   train_replay_sha256=train_meta["replay_sha256"], heldout_replay_sha256=heldout_meta["replay_sha256"],
                   train_outcomes=train_meta["status_counts"], heldout_outcomes=heldout_meta["status_counts"],
                   budget=dict(train_episodes=len(training.episodes), train_transitions=train_meta["env_steps"],
                               heldout_episodes=len(heldout.episodes), heldout_transitions=heldout_meta["env_steps"]),
                   passed=False, selected_recipe=None, attempts={})
    summary["source"] = _a1_reports(model, heldout, heldout_meta, source.mean, "source", out)
    for recipe in config["recipes"]:
        name = recipe["name"]
        # Both attempts start from exactly the same source weights and a fresh Adam.
        model.load_state_dict(saved["world_model"])
        opt = world_optimizer(model, recipe["lr"])
        start = time.monotonic()
        losses = fit_world_steps(model, opt, source, target, np.random.default_rng(config["fit_seed"]),
                                 updates=recipe["updates"], batch_size=config["batch_size"],
                                 source_fraction=recipe["source_fraction"])
        elapsed_s = time.monotonic() - start
        torch.save(dict(interface_spec=model.spec, wm_arch=saved["wm_arch"], world_model=model.state_dict(),
                        recipe=recipe, domain=domain, source_checkpoint_sha256=CHECKPOINT_SHA), out / f"{name}-wm.pt")
        scores = _a1_reports(model, heldout, heldout_meta, source.mean, name, out)
        summary["attempts"][name] = dict(recipe=recipe, scores=scores, mean_losses=losses, fit_seconds=elapsed_s)
        if scores["50"]["acceptance"]:
            summary.update(passed=True, selected_recipe=recipe)
            break
    _write_json(out / "summary.json", summary)
    return summary


@dataclass
class OnlineInputs:
    checkpoint: Path | str | dict
    prefill: Replay
    source_meta: dict
    validation: dict
    test: dict
    a1: dict
    domain_a1: dict | None = None


@dataclass
class InteractionBudget:
    train_episodes: int = 0
    train_transitions: int = 0
    cheap_episodes: int = 0
    heldout_episodes: int = 0
    full_episodes: int = 0

    def record(self, kind, episodes, transitions=0):
        if kind not in ("train", "cheap", "heldout", "full"):
            raise ValueError(f"unknown interaction category {kind}")
        if any(type(n) is not int or n < 0 for n in (episodes, transitions)):
            raise ValueError("interaction counts must be nonnegative integers")
        if kind != "train" and transitions:
            raise ValueError("only the training category tracks transitions")
        field = kind + "_episodes"
        setattr(self, field, getattr(self, field) + episodes)
        self.train_transitions += transitions


def select_checkpoint(cheap_rows):
    """The checkpoint selector consumes cheap-validation metrics alone."""
    return max(cheap_rows, key=nominal.validation_rank)


def _a1_settings(inputs, domain, smoke):
    if smoke:
        return 1e-4, 0.5
    a1 = inputs.a1
    recipe = a1.get("selected_recipe") or {}
    expected = {"main": (1e-4, .5, 1000), "fallback": (3e-4, .2, 3000)}
    if (a1.get("domain") != "P1" or a1.get("passed") is not True
            or recipe.get("name") not in expected
            or (recipe.get("lr"), recipe.get("source_fraction"), recipe.get("updates"))
            != expected[recipe["name"]]):
        raise ValueError("A2 requires a passing P1 A1 report under a prescribed recipe")
    domain_a1 = inputs.domain_a1 or a1
    if domain_a1["domain"] != domain:
        raise ValueError("A1 data provenance must refer to the online target domain")
    for report in (a1, domain_a1):
        if (report["source_checkpoint_sha256"] != file_hash(Path(inputs.checkpoint))
                or report["source_replay_sha256"] != inputs.source_meta["replay_sha256"]
                or len(report["train_reset_seeds"]) != 30 or len(report["heldout_reset_seeds"]) != 50):
            raise ValueError("A1 provenance differs from the frozen inputs or episode budget")
    return recipe["lr"], recipe["source_fraction"]


def _adapt_round(model, opt_wm, actor, critic, opt_actor, opt_critic, pools, near, world_rng, start_rng, config, fraction):
    losses = {}
    if opt_wm is not None:
        losses = fit_world_steps(model, opt_wm, *pools, world_rng, updates=config["wm_updates"],
                                 batch_size=config["wm_batch"], source_fraction=fraction)
    counts = _counts(config["imagined_batch"], fraction)
    averages = Counter()
    for _ in range(config["actor_updates"]):
        starts = torch.cat([_sample_starts(pool, strata, start_rng, n, config["near_fraction"])
                            for pool, strata, n in zip(pools, near, counts, strict=True)])
        feedback = update_actor_critic(
            model, actor, critic, starts, opt_actor, opt_critic, horizon=config["horizon"],
            gamma=config["gamma"], lam=config["lambda"], critic_epochs=config["critic_epochs"],
            actor_terminal_value=config["actor_terminal_value"],
            imagination_horizontal_cost_multiplier=config["imagination_horizontal_cost_multiplier"])
        for key, value in feedback.items():
            averages[key] += value / config["actor_updates"]
    return {**losses, **averages}


def _evaluate(policy, manifest, env_config, out):
    result = nominal.evaluate_nominal(policy, manifest, config=env_config)
    _write_json(out, result)
    metrics = {name: result[name] for name in
               ("n_episodes", "safe_landing_rate", "safe_in_target_rate", "mean_return", "status_counts")}
    metrics["safe_count"] = sum(ep["is_success"] for ep in result["episodes"])
    metrics["safe_in_target_count"] = sum(ep["safe_in_target"] for ep in result["episodes"])
    return metrics


def _run_recipe(config, recipe):
    if recipe not in ("main", "fallback"):
        raise ValueError(f"unknown A2 recipe {recipe}")
    settings = config.copy()
    if recipe == "fallback":
        settings["actor_lr"] *= config["fallback_actor_lr_multiplier"]
        settings["actor_updates"] = config["fallback_actor_updates"]
    return settings


def _g1(baseline):
    return (baseline["n_episodes"] == 100 and baseline["safe_count"] == 36
            and baseline["safe_in_target_count"] == 27
            and round(baseline["mean_return"], 4) == -36.3491)


def _cohort(paths, recipe):
    """Verify a completed three-seed P1 training cohort using validation results only."""
    if not paths or len(paths) != len(SEEDS):
        raise ValueError(f"three completed {recipe} P1 summary files are required")
    rows = [json.loads(Path(p).read_text()) for p in paths]
    if ({r["seed"] for r in rows} != set(SEEDS)
            or any(r["domain"] != "P1" or r["frozen"] or r["smoke"] or r["recipe"] != recipe
                   or not r["completed"] or not r["g1"] or r["budget"]["train_episodes"] != 150
                   or r["selected_full"]["n_episodes"] != 100 for r in rows)):
        raise ValueError(f"invalid P1 {recipe} validation cohort")
    return all(r["selected_full"]["safe_count"] >= 80 for r in rows)


def _authorize_online(domain, recipe, frozen, main_summaries, fallback_summaries):
    # P1 main and the identically configured frozen control can run together.
    if domain == "P1" and recipe == "main":
        return
    main_passed = _cohort(main_summaries, "main")
    if recipe == "fallback" and main_passed:
        raise ValueError("fallback is available only after P1 main misses G-A2")
    if domain == "P2":
        if frozen:
            raise ValueError("the P2 stress experiment uses the passing online model-adaptation recipe")
        if not main_passed and recipe == "fallback":
            if _cohort(fallback_summaries, "fallback"):
                return
        elif main_passed and recipe == "main":
            return
        raise ValueError("P2 requires the recipe that passed G-A2 on P1")


def run_online(inputs, config, out, *, domain="P1", seed=31, frozen=False, smoke=False,
               data_out=None, recipe="main", main_summaries=None, fallback_summaries=None):
    """Train one actor from v1; test scores are produced by a separate command after selection."""
    env_config = target_config(domain)
    if not smoke:
        _authorize_online(domain, recipe, frozen, main_summaries, fallback_summaries)
        if config != json.loads((Path(__file__).resolve().parents[2] / "configs/adapt_online.json").read_text()) or seed not in SEEDS:
            raise ValueError("formal online runs require the prescribed configuration and seed")
        verify_inputs(inputs.checkpoint, inputs.source_meta, inputs.validation, inputs.test)
    elif (config["iterations"] != 1 or config["collect_episodes"] != 2
          or not 0 <= config["wm_updates"] <= 1 or config["actor_updates"] != 1):
        raise ValueError("a smoke run uses one round, two resets and at most one of each update")
    c = _run_recipe(config, recipe)
    if smoke and recipe != "main":
        raise ValueError("smoke runs use the reduced main recipe")
    wm_lr, source_fraction = _a1_settings(inputs, domain, smoke)
    domain_a1 = inputs.domain_a1 or inputs.a1
    validation = nominal.manifest_prefix(inputs.validation, 2) if smoke else inputs.validation
    cheap = validation if smoke else nominal.cheap_validation_manifest(validation)
    # Precompute only reset identities, then check the actual returned IDs after each collection.
    namespace = {"P1": 500_000_000, "P2": 600_000_000}[domain]
    seeds = [namespace + seed * 100 + i for i in range(1, c["iterations"] + 1)]
    planned = [generate_initial_states(env_config, c["collect_episodes"], s, "train") for s in seeds]
    target_resets = domain_a1["train_reset_seeds"] + [
        s["seed"] for m in planned for s in m["starts"]]
    check_seed_partition(target_train=target_resets, target_heldout=domain_a1["heldout_reset_seeds"],
                         source_prefill=inputs.source_meta["reset_seeds"],
                         validation=[s["seed"] for s in inputs.validation["starts"]],
                         test=[s["seed"] for s in inputs.test["starts"]])

    out = Path(out)
    if data_out is not None and Path(data_out).exists():
        raise FileExistsError(f"online data directory already exists: {data_out}")
    out.mkdir(parents=True, exist_ok=False)
    if data_out is not None:
        Path(data_out).mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(c["cpu_threads"])
    torch.manual_seed(seed)
    # Frozen and adapted runs draw corresponding origins independently of model-update RNG consumption.
    world_rng, start_rng = [np.random.default_rng(stream) for stream in np.random.SeedSequence(seed).spawn(2)]
    policy, critic, model = initialize_models(inputs.checkpoint)
    actor = policy.actor
    opt_wm = None if frozen else world_optimizer(model, wm_lr)
    opt_actor = torch.optim.Adam(actor.parameters(), lr=c["actor_lr"])
    opt_critic = torch.optim.Adam(critic.parameters(), lr=c["critic_lr"])
    static = EpisodeWindows(inputs.prefill, inputs.source_meta, policy.spec, "cpu")
    near_static = _near(static, policy.spec, c)
    fifo = deque(maxlen=c["fifo_episodes"])
    records = deque(maxlen=c["fifo_episodes"])
    budget = InteractionBudget()
    cheap_rows, full_rows, curves, training = [], [], [], []
    provenance = dict(code=code_version(), runner_sha256=file_hash(Path(__file__)), domain=domain, seed=seed, recipe=recipe, frozen=frozen, smoke=smoke,
                      config=c, world_lr=wm_lr, source_fraction=source_fraction,
                      source_checkpoint_sha256=inputs.a1.get("source_checkpoint_sha256"),
                      source_replay_sha256=inputs.source_meta.get("replay_sha256"),
                      validation_sha256=inputs.validation["sha256"], test_sha256=inputs.test["sha256"],
                      selection_order=list(nominal.SELECTION_ORDER),
                      target_train_rollout_roots=seeds, cheap_manifest_sha256=cheap["sha256"],
                      cheap_start_ids=[s["id"] for s in cheap["starts"]],
                      a1_calibration_budget=domain_a1.get("budget", {}))
    _write_json(out / "config.json", provenance)
    began = time.monotonic()

    def validate(iteration):
        inexpensive = _evaluate(policy, cheap, env_config, out / f"cheap-{iteration:02d}.json")
        budget.record("cheap", inexpensive["n_episodes"])
        cheap_rows.append(dict(iteration=iteration, **inexpensive))
        selected = select_checkpoint(cheap_rows)["iteration"]
        if selected == iteration:
            staged = out / "best.tmp"
            torch.save(dict(policy_checkpoint(actor, policy.spec), iteration=iteration,
                            critic=critic.state_dict(), world_model=model.state_dict()), staged)
            staged.replace(out / "best.pt")
        # The selection and publication of best.pt are complete before full validation starts.
        full = _evaluate(policy, validation, env_config, out / f"full-{iteration:02d}.json")
        budget.record("full", full["n_episodes"])
        full_rows.append(dict(iteration=iteration, **full))
        curves.append(dict(iteration=iteration, train_episodes=budget.train_episodes,
                           train_transitions=budget.train_transitions, cheap=inexpensive, full=full,
                           cheap_selected_iteration=selected))
        _write_json(out / "learning-curve.json", curves)

    validate(0)
    zero_reproduced = _g1(full_rows[0]) if domain == "P1" and not smoke else None
    if zero_reproduced is False:
        _write_json(out / "summary.json", dict(provenance, completed=False, g1=False, budget=asdict(budget),
                                               baseline=full_rows[0]))
        raise ValueError("P1 round 0 does not reproduce 36 safe, 27 in target and return -36.3491")

    for iteration, (collect_seed, expected) in enumerate(zip(seeds, planned, strict=True), start=1):
        batch, metadata = nominal.collect_policy(policy, seed=collect_seed, episodes=c["collect_episodes"],
                                                 noise_std=c["noise_std"], config=env_config)
        if metadata["reset_seeds"] != [s["seed"] for s in expected["starts"]]:
            raise ValueError("target collection returned unexpected reset seeds")
        budget.record("train", len(batch.episodes), sum(map(len, batch.episodes)))
        if data_out is not None:
            adaptation = dict(domain=domain, seed=seed, iteration=iteration, recipe=recipe, frozen=frozen,
                              source_checkpoint_sha256=provenance["source_checkpoint_sha256"],
                              environment=env_config.to_dict())
            save_dataset(Path(data_out) / f"round-{iteration:02d}", batch, dict(metadata, adaptation=adaptation))
        fifo.extend(batch.episodes)
        records.extend(metadata["episode_provenance"])
        live = EpisodeWindows(Replay(list(fifo)), {"episode_provenance": list(records)}, policy.spec, "cpu")
        feedback = _adapt_round(model, opt_wm, actor, critic, opt_actor, opt_critic, (static, live),
                                (near_static, _near(live, policy.spec, c)), world_rng, start_rng, c, source_fraction)
        training.append(dict(iteration=iteration, collection_seed=collect_seed,
                             train_episodes=budget.train_episodes, train_transitions=budget.train_transitions,
                             collect_status_counts=metadata["status_counts"], fifo_episodes=len(fifo),
                             model_updates=0 if frozen else iteration * c["wm_updates"],
                             actor_updates=iteration * c["actor_updates"], elapsed_s=time.monotonic() - began,
                             **feedback))
        _write_json(out / "training.json", training)
        if iteration % c["validate_every"] == 0 or iteration == c["iterations"]:
            validate(iteration)

    chosen = select_checkpoint(cheap_rows)
    full_best = max(full_rows, key=nominal.validation_rank)
    chosen_full = next(row for row in full_rows if row["iteration"] == chosen["iteration"])
    result = dict(provenance, completed=True, g1=zero_reproduced, budget=asdict(budget),
                  selected_cheap=chosen, selected_full=chosen_full, full_oracle=full_best,
                  selected_checkpoint_sha256=file_hash(out / "best.pt"),
                  selected_full_at_least_80=chosen_full["safe_count"] >= 80 if not smoke else None,
                  model_updates=training[-1]["model_updates"], actor_updates=training[-1]["actor_updates"],
                  elapsed_s=time.monotonic() - began)
    _write_json(out / "summary.json", result)
    return result
