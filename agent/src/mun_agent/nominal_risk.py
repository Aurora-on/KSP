"""Validation-only diagnostics for the selected nominal policy and each observed safety decline."""

import json
from pathlib import Path

from .dataset import file_hash, load_dataset
from .fit import computation_device
from .nominal import load_manifest, policy_risk, risk_report, visited_pairs
from .nominal_report import read_candidate, save_json, validation_rank
from .policy import load_policy
from .train import load_training_world_model


def add_risk_command(commands):
    parser = commands.add_parser("policy-risk", help="measure nominal policy and safety-decline risks on validation starts")
    parser.add_argument("--run", type=Path, action="append", required=True, help="completed seed run (repeatable)")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True, help="permanent training prefill")
    parser.add_argument("--out", type=Path, required=True, help="output directory for the index, assessments and cached points")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda", "auto"), default="cpu")
    parser.set_defaults(handler=run_risk_command)


def _point_name(row):
    return "bc-only" if row["phase"] == "bc_only" else f"online-{row['iteration']:06d}"


def declines(rows):
    points = [r for r in rows if r.get("phase") in ("bc_only", "online")]
    if points and (points[0]["phase"] != "bc_only" or points[0]["iteration"] != 0):
        raise ValueError("nominal validation chronology needs the cloning baseline")
    pairs = list(zip(points, points[1:]))
    for before, after in pairs:
        if after["phase"] != "online" or after["iteration"] <= before["iteration"]:
            raise ValueError("online validation iterations must increase strictly")
    return [(before, after) for before, after in pairs if before["safe_landing_rate"] > after["safe_landing_rate"]]


def _verify_visits(visited, official, manifest):
    """Match every stochasticity-free reroll to its stored training observation."""
    if official["starts_sha256"] != manifest["sha256"]:
        raise ValueError("archived evaluation used different validation starts")
    observed = [(r["start_id"], r["status"]) for r in visited["first_touchdowns"]]
    expected = [(r["start_id"], r["status"]) for r in official["episodes"]]
    if expected != observed or [i for i, _ in observed] != [s["id"] for s in manifest["starts"]]:
        raise ValueError("risk trajectories differ from archived validation episode identities or outcomes")


def _cache_point(run, seed, row, manifest, out, device, *, visited=None):
    name = _point_name(row)
    checkpoint = run / "checkpoints" / (name + ".pt")
    official = run / "validation" / (name + ".json")
    destination = out / "points" / f"seed-{seed}-{name}.json"
    identity = {"schema": 1, "checkpoint_sha256": file_hash(checkpoint), "official_sha256": file_hash(official),
                "validation_sha256": manifest["sha256"], "phase": row["phase"], "iteration": row["iteration"],
                "seed": seed, "device": str(device)}
    if destination.exists():
        existing = json.loads(destination.read_text())
        if existing.get("identity") == identity:
            return str(destination.relative_to(out))
    if visited is None:
        model = load_training_world_model(checkpoint, device)
        visited = policy_risk(model, load_policy(checkpoint, device), manifest)
    official_result = json.loads(official.read_text())
    _verify_visits(visited, official_result, manifest)
    if official_result["safe_landing_rate"] != row["safe_landing_rate"]:
        raise ValueError("validation log and archived rates disagree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_json(destination, dict(identity=identity, safe_landing_rate=row["safe_landing_rate"], visited=visited))
    return str(destination.relative_to(out))


def _drop_records(run, seed, manifest, out, device, *, shared=None):
    rows = [json.loads(line) for line in (run / "validation.jsonl").read_text().splitlines() if line]
    records = []
    for before, after in declines(rows):
        files = []
        for row in (before, after):
            if row["starts_sha256"] != manifest["sha256"]:
                raise ValueError("drop point was evaluated on a different validation manifest")
            visited = (shared or {}).get(_point_name(row))
            files.append(_cache_point(run, seed, row, manifest, out, device, visited=visited))
        records.append({"before": files[0], "after": files[1], "before_safe_rate": before["safe_landing_rate"],
                        "after_safe_rate": after["safe_landing_rate"], "before_iteration": before["iteration"],
                        "after_iteration": after["iteration"]})
    return records


def run_risk_command(args):
    import torch

    manifest = load_manifest(args.validation, "validation")
    device = computation_device(args.device)
    candidates = sorted((read_candidate(run, manifest["sha256"]) for run in args.run), key=lambda c: c["seed"])
    if not candidates or len({c["seed"] for c in candidates}) != len(candidates):
        raise ValueError("risk reporting requires distinct completed seeds")
    selected = max(candidates, key=lambda c: validation_rank(c["best"]))
    replay, metadata = load_dataset(args.data)
    if any(c["prefill_replay_sha256"] != metadata["replay_sha256"] for c in candidates):
        raise ValueError("risk prefill must match every run's permanent training source")

    run = Path(selected["run"])
    best = run / "checkpoints/best.pt"
    point = _point_name(selected["best"])
    frozen = run / "checkpoints" / (point + ".pt")
    # The winner's full training checkpoint and its smaller validation snapshot
    # differ in storage format. Reuse real rollouts only for identical inference weights.
    saved = torch.load(best, map_location="cpu", weights_only=True)
    snapshot = torch.load(frozen, map_location="cpu", weights_only=True)
    for key in ("actor", "world_model"):
        if saved[key].keys() != snapshot[key].keys() or any(
                not torch.equal(v, snapshot[key][k]) for k, v in saved[key].items()):
            raise ValueError(f"selected {key} differs from its validation snapshot")
    identity = {"schema": 1, "seed": selected["seed"], "iteration": selected["best"]["iteration"],
                "checkpoint_sha256": file_hash(best), "validation_sha256": manifest["sha256"],
                "prefill_replay_sha256": metadata["replay_sha256"], "device": str(device)}
    args.out.mkdir(parents=True, exist_ok=True)
    selected_file = args.out / "selected-policy.json"
    winner = json.loads(selected_file.read_text()) if selected_file.exists() else {}
    if winner.get("identity") != identity:
        model = load_training_world_model(saved, device)
        policy = load_policy(best, device)
        winner = risk_report(model, policy.actor, policy, manifest, replay, metadata["episode_provenance"])
        winner["identity"] = identity
        official = json.loads((run / "validation" / (point + ".json")).read_text())
        _verify_visits(winner["policy_visited"], official, manifest)
        save_json(selected_file, winner)
    last_point = _point_name(selected["last"])
    last_file = selected_file
    last = winner
    if last_point != point:
        last_file = args.out / "last-policy-selected-seed.json"
        last_checkpoint = run / "checkpoints" / (last_point + ".pt")
        last_id = dict(identity, iteration=selected["last"]["iteration"], checkpoint_sha256=file_hash(last_checkpoint))
        last = json.loads(last_file.read_text()) if last_file.exists() else {}
        if last.get("identity") != last_id:
            model = load_training_world_model(last_checkpoint, device)
            policy = load_policy(last_checkpoint, device)
            last = visited_pairs(model, policy.actor, policy, manifest)
            last["identity"] = last_id
            official = json.loads((run / "validation" / (last_point + ".json")).read_text())
            _verify_visits(last["policy_visited"], official, manifest)
            save_json(last_file, last)
    winner_seed_visits = {point: winner["policy_visited"], last_point: last["policy_visited"]}
    runs = []
    for candidate in candidates:
        shared = winner_seed_visits if candidate["seed"] == selected["seed"] else None
        records = _drop_records(Path(candidate["run"]), candidate["seed"], manifest, args.out, device, shared=shared)
        runs.append({"seed": candidate["seed"], "run": candidate["run"], "declines": records})
    report = {"validation_sha256": manifest["sha256"], "validation_episodes": len(manifest["starts"]),
              "selected": identity, "selected_policy": selected_file.name, "runs": runs,
              "last_policy_selected_seed": last_file.name, "best_validation": selected["best"],
              "last_validation_selected_seed": selected["last"],
              "declines": sum(len(r["declines"]) for r in runs),
              "definition": "Every decrease in safe landing rate between adjacent scheduled validation evaluations starting from the BC-only checkpoint. Before and after use the same validation starts, with each snapshot's own model and deterministic policy."}
    save_json(args.out / "index.json", report)
    return dict(out=str(args.out), selected_seed=selected["seed"], iteration=identity["iteration"],
                declines=report["declines"], starts_sha256=manifest["sha256"])
