"""Compare frozen critic changes with the actor's paired real-simulator returns on train starts."""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
from mun_lander import LanderEnv, SimConfig
from mun_lander.evaluation import generate_initial_states

from mun_agent.dataset import file_hash
from mun_agent.imagination import Critic
from mun_agent.interface import to_native_action
from mun_agent.policy import load_policy


def snapshots(env, policy, start):
    obs, _ = env.reset(seed=start["seed"], options={"initial_state": start["initial_state"]})
    policy.reset()
    found = {}
    while True:
        phases = {"8s": env.time >= 8, "below_10m": obs[2] < 10, "below_2m": obs[2] < 2,
                  "60s": env.time >= 60, "80s": env.time >= 80}
        if not env.contact and obs[2] >= policy.spec["ground_epsilon_m"]:
            for phase, due in phases.items():
                if due and phase not in found:
                    found[phase] = env.get_state()
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        if terminated or truncated:
            return found


def branch(env, policy, critic, snapshot, gamma, extra_cost):
    env.set_state(snapshot)
    policy.reset()
    policy.steps = snapshot["steps"]
    # Every snapshot is airborne. The first state's ground-history counter is therefore zero.
    state = torch.from_numpy(policy.state(env.state.observation()))
    value = float(critic(state))
    action = to_native_action(policy.actor(state).numpy())
    raw_return = target_return = 0.
    discount = 1.
    while True:
        obs, reward, terminated, truncated, info = env.step(action)
        raw_return += discount * reward
        target_return += discount * (reward - extra_cost * info["horizontal_error"] * env.config.control_dt)
        if terminated or truncated:
            return value, raw_return, target_return, info["status"]
        discount *= gamma
        action = policy(obs)


def perturb(snapshot, axis, delta, dt):
    changed = copy.deepcopy(snapshot)
    if axis == "clock_s":
        changed["time"] += delta
        changed["steps"] += round(delta / dt)
    elif axis == "vz_m_s":
        changed["state"]["velocity"][2] += delta
    else:
        changed["state"]["position"][0 if axis == "x_m" else 1] += delta
    return changed


def summarize(records, tolerance):
    grouped = defaultdict(list)
    for r in records:
        grouped[r["phase"], r["axis"], r["delta"]].append(r)
    summaries = []
    for (phase, axis, delta), rows in grouped.items():
        true = np.array([r["true_delta"] for r in rows])
        predicted = np.array([r["critic_delta"] for r in rows])
        informative = np.abs(true) > tolerance
        agreements = np.sign(true[informative]) == np.sign(predicted[informative])
        summaries.append(dict(phase=phase, axis=axis, delta=delta, n=len(rows),
                              mean_true_delta=float(true.mean()), mean_critic_delta=float(predicted.mean()),
                              mean_raw_delta=float(np.mean([r["raw_delta"] for r in rows])),
                              mean_abs_delta_error=float(np.abs(predicted - true).mean()),
                              small_true_deltas=int((~informative).sum()), sign_agreements=int(agreements.sum()),
                              sign_agreement_rate=float(agreements.mean()) if len(agreements) else None,
                              outcome_changes=sum(r["base_status"] != r["status"] for r in rows)))
    return summaries


@torch.inference_mode()
def run(args):
    policy = load_policy(args.actor)
    actor_saved = torch.load(args.actor, map_location="cpu", weights_only=True)
    saved = torch.load(args.critic, map_location="cpu", weights_only=True)
    if saved.get("interface_spec", policy.spec) != policy.spec:
        raise ValueError("actor and critic use different observation specifications")
    # A standalone weight-only critic uses the architecture and reward contract of the supplied actor's run.
    settings = saved.get("provenance", actor_saved.get("provenance", {})).get("config")
    if settings is None:
        raise ValueError("one checkpoint must contain the critic architecture, reward settings and discount")
    critic = Critic(settings["critic_width"], settings["value_scale"], settings.get("critic_upper_bound"),
                    settings.get("yaw_invariant", False)).eval()
    critic.load_state_dict(saved["critic"])
    gamma = settings["gamma"]
    multiplier = settings.get("imagination_horizontal_cost_multiplier", 1.)
    config = SimConfig()
    extra_cost = (multiplier - 1.) * config.reward.horizontal_cost
    dt = config.control_dt
    if not math.isclose(args.clock / dt, round(args.clock / dt), abs_tol=1e-8):
        raise ValueError("clock perturbation must be an integer number of control steps")
    plans = [(axis, sign * size) for axis, size in (("vz_m_s", args.vz), ("x_m", args.xy), ("y_m", args.xy))
             for sign in (1, -1)]
    plans.append(("clock_s", args.clock))
    manifest = generate_initial_states(config, args.episodes, args.start_seed, "train")
    records, skipped = [], []
    env = LanderEnv(config)
    try:
        for start in manifest["starts"]:
            for phase, original in snapshots(env, policy, start).items():
                base_v, base_raw, base_g, base_status = branch(env, policy, critic, original, gamma, extra_cost)
                for axis, delta in plans:
                    changed = perturb(original, axis, delta, dt)
                    ids = dict(start_id=start["id"], phase=phase, axis=axis, delta=delta)
                    if changed["time"] >= config.max_time:
                        skipped.append(dict(**ids, reason="perturbed start reaches the episode time limit"))
                        continue
                    v, raw, g, status = branch(env, policy, critic, changed, gamma, extra_cost)
                    records.append(dict(**ids, time_s=original["time"], height_m=original["state"]["position"][2],
                                        base_status=base_status, status=status, base_v=base_v, base_g=base_g,
                                        base_raw_return=base_raw, critic_delta=v-base_v, true_delta=g-base_g,
                                        raw_delta=raw-base_raw))
    finally:
        env.close()
    return dict(split="train", start_seed=args.start_seed, episodes=args.episodes, train_manifest_sha256=manifest["sha256"],
                actor_sha256=file_hash(args.actor), critic_sha256=file_hash(args.critic), gamma=gamma,
                imagination_horizontal_cost_multiplier=multiplier, sign_tolerance=args.sign_tolerance,
                definition="true_delta compares discounted to-termination returns under the same frozen actor. "
                           "raw_delta uses only simulator rewards; true_delta also applies the critic's configured "
                           "training-only horizontal cost to real successor distances, using its nominal control dt. "
                           "Sign rates exclude true differences within the requested tolerance. Phases use the first "
                           "eligible airborne state. Late phases are absent for episodes ending earlier.",
                groups=summarize(records, args.sign_tolerance), skipped=skipped, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--critic", type=Path, required=True)
    parser.add_argument("--start-seed", type=int, required=True, help="root seed for fresh nominal train starts")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--vz", type=float, default=.5, help="two-sided vertical velocity perturbation in m/s")
    parser.add_argument("--xy", type=float, default=3., help="two-sided per-axis horizontal displacement in m")
    parser.add_argument("--clock", type=float, default=20., help="forward clock perturbation in s")
    parser.add_argument("--sign-tolerance", type=float, default=0., help="exclude near-zero real changes when counting signs")
    args = parser.parse_args()
    if (args.start_seed < 0 or args.episodes < 1 or not math.isfinite(args.sign_tolerance) or args.sign_tolerance < 0
            or not all(math.isfinite(x) and x > 0 for x in (args.vz, args.xy, args.clock))):
        parser.error("expected nonnegative seed/tolerance, positive episode count and positive finite perturbations")
    torch.set_num_threads(1)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"{len(result['records'])} paired perturbations across {args.episodes} train starts -> {args.out}")


if __name__ == "__main__":
    main()
