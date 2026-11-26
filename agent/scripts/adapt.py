"""Offline calibration and bounded, simulator-to-simulator online adaptation on CPU."""
import argparse
import json
from pathlib import Path

import torch

from mun_agent import nominal
from mun_agent.adapt import CHECKPOINT_SHA, MANIFEST_SHA, PREFILL_SHA, OnlineInputs, SEEDS, _cohort, _evaluate, _g1, run_a1, run_online
from mun_agent.dataset import file_hash, load_dataset
from mun_agent.policy import load_policy
from mun_agent.train import _write_json
from mun_agent.transfer import target_config


AGENT = Path(__file__).resolve().parents[1]
RUNS = AGENT / "runs" / "adapt"
DATA = AGENT / "data"
CHECKPOINT = AGENT / "runs" / "s3-stable" / "seed-33" / "checkpoints" / "best.pt"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_selected_test(out, test_path, main_summaries, fallback_summaries=None):
    """Test the locked P1 recipe once, including a final failed fallback if both recipes missed."""
    out = Path(out)
    result = read_json(out / "summary.json")
    if result["domain"] != "P1" or result["smoke"] or not result["completed"] or not result["g1"]:
        raise ValueError("final test requires a complete formal P1 online run")
    main_passed = _cohort(main_summaries, "main")
    final_recipe = "main" if main_passed else "fallback"
    if final_recipe == "fallback":
        _cohort(fallback_summaries, "fallback")
    if result["recipe"] != final_recipe:
        raise ValueError("test selection must follow the final validation-only recipe decision")
    if not result["frozen"]:
        cohort = main_summaries if main_passed else fallback_summaries
        if (out / "summary.json").resolve() not in {Path(p).resolve() for p in cohort}:
            raise ValueError("online test run must belong to the locked P1 cohort")
    manifest = nominal.load_manifest(test_path, "test")
    if manifest["sha256"] != MANIFEST_SHA["test"] or manifest["sha256"] != result["test_sha256"]:
        raise ValueError("final test requires the frozen 100-entry nominal test manifest")
    checkpoint = out / "best.pt"
    if file_hash(checkpoint) != result["selected_checkpoint_sha256"]:
        raise ValueError("cheap-selected checkpoint was replaced")
    policy = load_policy(checkpoint)
    torch.set_num_threads(result["config"]["cpu_threads"])
    # Reserve the evaluation before interacting. A partial file survives interruption.
    with (out / "test.json").open("x", encoding="utf-8") as reservation:
        json.dump({"completed": False, "checkpoint_sha256": result["selected_checkpoint_sha256"]}, reservation)
    metrics = _evaluate(policy, manifest, target_config("P1"), out / "test-episodes.json")
    receipt = dict(completed=True, domain="P1", seed=result["seed"], recipe=result["recipe"],
                   frozen=result["frozen"], selected_iteration=result["selected_cheap"]["iteration"],
                   checkpoint_sha256=result["selected_checkpoint_sha256"], test_sha256=manifest["sha256"],
                   test_episodes=metrics["n_episodes"], **metrics)
    _write_json(out / "test.json", receipt)
    return receipt


def _inputs_options(parser):
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--prefill", type=Path, default=DATA / "s3-nominal-prefill")
    parser.add_argument("--validation", type=Path, default=DATA / "s3-validation.json")
    parser.add_argument("--test", type=Path, default=DATA / "s3-test.json")
    parser.add_argument("--domain", required=True, choices=("P1", "P2"))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DATA)


def _cohort_options(parser):
    parser.add_argument("--p1-main", type=Path, nargs=3, metavar="SUMMARY",
                        help="three P1/main summary.json files, one per seed")
    parser.add_argument("--p1-fallback", type=Path, nargs=3, metavar="SUMMARY",
                        help="three P1/fallback summary.json files, one per seed")


def run_c(out, main_summaries, fallback_summaries, resume=False):
    """Repeat source S3 with only the first-contact imagination penalty changed."""
    from dataclasses import asdict
    from mun_agent.imagination import TouchdownMargin, imagined_touchdown_margin
    from mun_agent.train import train_policy
    from transfer_sensitivity import conditions, evaluate_pair, summarize, IDENTITIES

    if _cohort(main_summaries, "main"):
        passed = main_summaries
    elif _cohort(fallback_summaries, "fallback"):
        passed = fallback_summaries
    else:
        raise ValueError("C requires a passing three-seed P1 cohort")
    prefill = DATA / "s3-nominal-prefill"
    validation = DATA / "s3-validation.json"
    model_path = AGENT / "runs/s1-contact-seed31/best.pt"
    bc = CHECKPOINT.with_name("bc-only.pt")
    base = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    config = read_json(AGENT / "configs/s3_nominal.json")
    manifest = nominal.load_manifest(validation, "validation")
    _, meta = load_dataset(prefill)
    if (file_hash(CHECKPOINT) != CHECKPOINT_SHA or config != base["provenance"]["config"]
            or file_hash(bc) != IDENTITIES["bc_only"]
            or meta["replay_sha256"] != PREFILL_SHA or manifest["sha256"] != MANIFEST_SHA["validation"]
            or file_hash(model_path) != base["provenance"]["pretrained_sha256"]):
        raise ValueError("C requires the original S3 seed-33 recipe, S1 model, prefill and validation starts")
    margin = TouchdownMargin(speed_m_s=1., cost_per_m_s=10.)
    config = dict(config, imagined_first_touchdown_margin=asdict(margin))
    with imagined_touchdown_margin(margin):
        training = train_policy(config, prefill, model_path, validation, out, seed=33, device="cpu", resume=resume)
    if not training["completed_fixed_iterations"]:
        raise ValueError("C did not complete its prescribed 24 rounds")

    selected = out / "checkpoints/best.pt"
    catalog = conditions()
    results = {}
    for name in ("nominal", "inertia_2.5", "inertia_3", "inertia_4"):
        results[name] = {}
        for label, controller, checkpoint in (("C", "online", selected), ("v1", "online", CHECKPOINT),
                                              ("BC", "bc_only", bc)):
            episodes = evaluate_pair(catalog[name], controller, manifest["starts"], checkpoint, bc)
            _write_json(out / f"{name}-{label}.json", episodes)
            results[name][label] = summarize(episodes)
    impact = results["nominal"]["C"]["mean_touchdown_speed_m_s"]
    decision = dict(nominal_precision=(results["nominal"]["C"]["safe_count"] >= 99
                                       and results["nominal"]["C"]["safe_in_target_count"] >= 95),
                    nominal_impact_speed=impact is not None and impact <= 1.,
                    inertia_transfer=(results["inertia_3"]["C"]["safe_count"] >= 86
                                      and results["inertia_4"]["C"]["safe_count"] >= 46))
    _write_json(out / "decision.json", dict(seed=33, margin=asdict(margin), best_iteration=training["best_iteration"],
                selected_checkpoint_sha256=file_hash(selected), validation_sha256=manifest["sha256"],
                p1_gate=[dict(path=str(p), sha256=file_hash(p)) for p in passed],
                criteria=decision, recommend=all(decision.values()), results=results))


def run_g1(checkpoint, validation_path, destination):
    """Independently reproduce the frozen P1 start before any online training."""
    manifest = nominal.load_manifest(validation_path, "validation")
    if file_hash(checkpoint) != CHECKPOINT_SHA or manifest["sha256"] != MANIFEST_SHA["validation"]:
        raise ValueError("G1 requires the registered policy and validation manifest")
    torch.set_num_threads(1)
    result = nominal.evaluate_nominal(load_policy(checkpoint), manifest, config=target_config("P1"))
    safe = sum(ep["is_success"] for ep in result["episodes"])
    target = sum(ep["safe_in_target"] for ep in result["episodes"])
    passed = _g1(dict(result, safe_count=safe, safe_in_target_count=target))
    destination.parent.mkdir(exist_ok=True, parents=True)
    _write_json(destination, {"domain": "P1", "split": "validation", "device": "cpu",
                              "episodes": result["n_episodes"], "checkpoint_sha256": CHECKPOINT_SHA,
                              "manifest_sha256": manifest["sha256"], "validation_file_sha256": file_hash(validation_path),
                              "safe_count": safe, "safe_in_target_count": target,
                              "mean_return": result["mean_return"], "G1_passed": passed})
    if not passed:
        raise ValueError("P1 zero-shot result differs from the registered baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    baseline = commands.add_parser("g1", help="reproduce the frozen policy on P1 validation starts")
    baseline.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    baseline.add_argument("--validation", type=Path, default=DATA / "s3-validation.json")
    baseline.add_argument("--out", type=Path, default=RUNS / "g1.json")
    offline = commands.add_parser("a1", help="calibrate the world model on 30 target episodes")
    _inputs_options(offline)
    offline.add_argument("--config", type=Path, default=AGENT / "configs" / "adapt_a1.json")
    online = commands.add_parser("a2", help="run one formal online adaptation or frozen control")
    _inputs_options(online)
    _cohort_options(online)
    online.add_argument("--config", type=Path, default=AGENT / "configs" / "adapt_online.json")
    online.add_argument("--a1", type=Path, default=RUNS / "a1" / "P1" / "summary.json")
    online.add_argument("--domain-a1", type=Path, help="A1 provenance for P2 (uses the P1 recipe)")
    online.add_argument("--variant", choices=("main", "fallback", "frozen"), default="main")
    online.add_argument("--recipe", choices=("main", "fallback"), help="recipe matched by the frozen control")
    online.add_argument("--seed", required=True, type=int, choices=SEEDS)
    test = commands.add_parser("test", help="evaluate a final cheap-selected P1 checkpoint once")
    test.add_argument("--out", required=True, type=Path, help="the completed selected run directory")
    test.add_argument("--test", type=Path, default=DATA / "s3-test.json")
    _cohort_options(test)
    margin = commands.add_parser("c", help="train the gated source-domain first-touchdown margin experiment")
    _cohort_options(margin)
    margin.add_argument("--out", default=RUNS / "c", type=Path)
    margin.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "c":
        run_c(args.out, args.p1_main, args.p1_fallback, args.resume)
        return
    if args.stage == "g1":
        run_g1(args.checkpoint, args.validation, args.out)
        return
    if args.stage == "test":
        run_selected_test(args.out, args.test, args.p1_main, args.p1_fallback)
        return
    if args.stage == "a1":
        run_a1(args.domain, read_json(args.config), checkpoint=args.checkpoint, prefill_path=args.prefill,
               data_dir=args.data_dir, out=args.out or RUNS / "a1" / args.domain,
               validation_path=args.validation, test_path=args.test)
        return

    frozen = args.variant == "frozen"
    if args.recipe and not frozen:
        parser.error("--recipe applies only to --variant frozen")
    recipe = args.recipe or ("main" if frozen else args.variant)
    a1 = read_json(args.a1)
    domain_a1 = a1
    if args.domain == "P2":
        domain_a1 = read_json(args.domain_a1 or RUNS / "a1" / "P2" / "summary.json")
    prefill, source_meta = load_dataset(args.prefill)
    inputs = OnlineInputs(args.checkpoint, prefill, source_meta,
                          nominal.load_manifest(args.validation, "validation"),
                          nominal.load_manifest(args.test, "test"), a1, domain_a1)
    group = ("frozen-" if frozen else "online-") + recipe
    out = args.out or RUNS / "a2" / args.domain / group / f"seed-{args.seed}"
    data_out = args.data_dir / f"adapt-a2-{args.domain}-{group}-seed-{args.seed}"
    run_online(inputs, read_json(args.config), out, domain=args.domain, seed=args.seed,
               frozen=frozen, data_out=data_out, recipe=recipe, main_summaries=args.p1_main,
               fallback_summaries=args.p1_fallback)


if __name__ == "__main__":
    main()
