# KSP replay handoff (draft)

The world-model team can read complete simulator episodes today. The KSP adapter team can start by sharing a time-aligned raw trace and the frame and control definitions below. KSP observations, rewards and outcomes still need joint calibration. The recorded Mun orbit probe verified a named-save reset and a 0.02 s step. That step returned `reward=None`, both ending flags false and `episode_evaluated=False`. [Adapter](../ksp_env/env.py) · [StepResult](../ksp_env/contracts.py) · [Live validation](validation.md)

## What the current reader accepts

For an episode with `L > 0` transitions, row `i` describes the applied action from `obs[i]` to `obs[i+1]`. Capture `obs[0]` from the verified reset and `obs[L]` from the final step, including a truncated step. The next reset starts its own observation block. [Collector](../agent/src/mun_agent/collect.py) · [Episode and Replay](../agent/src/mun_agent/replay.py)

| Replay field | Per episode | Native simulator meaning |
|---|---|---|
| `obs` | float32 `[L+1,14]` | Position in m (0–2), velocity in m/s (3–5), unit body-to-world quaternion `wxyz` (6–9), body angular velocity in rad/s (10–12), main fuel in kg (13). The target-local world frame is right-handed with +z up. |
| `action` | float32 `[L,4]` | Applied `[throttle, rcs_x, rcs_y, rcs_z]` after clipping. Throttle is in [0,1], the three normalized body torque commands in [−1,1]. The policy's internal symmetric throttle is a training transform. |
| `reward` | float32 `[L]` | Finite scalar for the same transition, including the final outcome contribution. KSP needs a versioned reward definition before producing training rows. |
| `terminated`, `truncated` | bool `[L]` each | Both false before the end. A labeled final transition has exactly one true flag, according to the outcome table below. |
| `dt` | float32 `[L]` | Actual elapsed simulation seconds for each applied action. The reader accepts finite, nonnegative values and legacy episodes with no duration field. Record measured duration for every new episode. |

For `N` episodes and `T = sum(lengths)`, `replay.npz` stores `lengths` as int64 `[N]`, `obs` as `[T+N,14]`, `action` as `[T,4]` and each scalar transition field as `[T]`. A mixed legacy archive can carry a boolean `episode_has_dt[N]` mask. Outcomes live in `collection.json` under `episode_provenance[i].status`. The collector also records first-touchdown speed and tilt when available. The deterministic learner derives 18 features from the raw observation history. Per-tick contact and touchdown events are absent from the NPZ. [Replay I/O](../agent/src/mun_agent/replay.py) · [State transforms](../agent/src/mun_agent/interface.py)

`load_dataset(directory)` verifies the NPZ SHA-256 and an embedded simulator **train** manifest and its hash. It compares every initial observation exactly to its manifest initial state and checks available provenance against the episode IDs, seeds, lengths, outcomes and flags. A KSP archive needs its own provenance reader or explicit conversion contract that preserves KSP split, save identity and reset observations. For the first exchange, deliver raw records with source and split identified. [Dataset reader and writer](../agent/src/mun_agent/dataset.py) · [KSP splits](../configs/mun_landing.toml)

## Controls, coordinates and time

The shared KSP action order is `[rcs_pitch, rcs_yaw, rcs_roll, main_engine_throttle]`. The test adapter writes named kRPC normalized controls. Moving throttle to slot 0 resolves only the field order. To determine axis signs, gains and delays, share the craft's reference part, an upright rest sample, signed single-axis pulse traces and a vertical engine pulse. Record requested and applied controls, their timestamps and the resulting body rates or thrust. [Shared contract](task-spec.md) · [Simulator integration notes](../simulator/docs/REPOSITORY_INTEGRATION.md)

Supply the target origin, basis and handedness, the craft basis, the attitude transform direction and the fuel resource-to-kg conversion. kRPC currently reads `xyzw`, position, velocity and angular velocity in the selected target frame, plus a resource amount. Converting the quaternion components alone leaves the basis and angular-velocity frame unresolved. Identify the position reference point and its offset from deployed gear on the local terrain. [KSP observation](../ksp_env/env.py) · [Simulator state](../simulator/src/mun_lander/state.py)

Give each control interval its measured start/end universal time (UT) and each raw physics sample its UT. For grouped ticks, `dt = ut_end_s - ut_start_s` over the whole action hold. Record skipped ticks, tick or load failures and reload boundaries explicitly. The simulator uses a nominal 0.1 s control interval and can end a transition early. The KSP task interval is open. [Adapter timing](../ksp_env/env.py) · [Task timing](task-spec.md)

## Outcomes and contact calibration

The labels below are the world-model vocabulary. The KSP evaluator still needs an agreed mapping. Its initial time-stamped evidence should retain first contact, subsequent contacts and the physical reason for ending each attempt. [Simulator event handling](../simulator/src/mun_lander/env.py) · [Classifier labels](../agent/src/mun_agent/deterministic_fit.py)

| World-model label | terminated / truncated | Interpretation |
|---|---|---|
| `continue` | false / false | Flying or stabilizing, including an acceptable first touch before 5 s. |
| `safe_landing` | true / false | Safe touchdown followed by 5 continuous seconds of stable contact. KSP also requires vessel survival and integrity of the relevant parts. |
| `crash` | true / false | Source simulator detects unsafe motion on impact or during contact. KSP needs agreed impact, stability and damage criteria. |
| `timeout` | false / true | Physical episode time limit reached. |
| `out_of_bounds` | false / true | Source simulator position limit exceeded. KSP bounds and adoption of this outcome remain open. |
| `fuel_exhausted` | true / false | Source simulator aborts for exhausted main fuel while airborne. Proposed mapping from shared `fuel_exhaustion_abort`. |

The source classifier learns continue/safe/crash and masks the final transition of the three rule-based aborts. An infrastructure incident (`connection_loss`, `reset_failure` or `tick_timeout`) belongs in an attempt log with the last valid UT and error, outside the completed physical outcomes. Keep a partial trace for diagnosis. [Shared failure contract](../configs/mun_landing.toml) · [Training labels](../agent/src/mun_agent/deterministic_fit.py)

The source model's `z < 3e-6 m` history feature was calibrated on simulator trajectories. For KSP, compare `Vessel.situation`, terrain-relative clearance corrected for craft/gear geometry, vertical and horizontal speeds, tilt, angular rate and part integrity at matched UTs. Include direct gear-contact telemetry if available and identify missing signals. Label impact from the last pre-impact sample or bracket it between raw ticks, then follow settlement. Compare an upright stationary craft, low hover, first touch, rebound and recontact to measure offsets, noise and delays before choosing a tolerance. Keep these diagnostic signals alongside the raw trace. [Simulator history](../agent/src/mun_agent/interface.py) · [KSP telemetry](ksp-env.md) · [Task safety criteria](task-spec.md)

The 5 s requirement uses continuous **game time**. Proposed KSP handling resets the stability clock at episode reset, loss of contact or loss of stability. A later valid contact starts a fresh window. A sampling gap leaves continuous support unverified. The simulator evaluates impact and stability before the post-substep bounds and airborne-fuel checks, with timeout last. It also checks for already exhausted fuel before advancing an airborne craft. For KSP, preserve competing event flags and UT brackets, including destruction versus contact and contact versus timeout or depletion. Agree how to resolve simultaneous or ambiguous events using paired examples before assigning final labels. [Simulator event order](../simulator/src/mun_lander/env.py)

## First exchange

Please return a small JSONL or CSV example with a field dictionary and units. Include the craft/save identity and revision, target/frame definitions, split, attempt ID, reset metadata, timing and control channels. One complete safe trace should span the reset, approach, first contact and 5 s confirmation with its final observation. Add short labeled excerpts for hard impact, rebound/recontact and a failed or interrupted tick as available. Mark unavailable cases and signals explicitly. For each completed attempt provide the outcome, reason, final flags and reward definition or pending status. These examples let both teams check row alignment, axis response, contact timing and failure handling before conversion.

## Reproduce the offline reader check

Run from the repository root with the simulator and agent installed. This creates a throwaway two-episode collection and checks all six arrays after the actual dataset writer/reader round trip.

```sh
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from mun_agent.collect import collect
from mun_agent.dataset import load_dataset, save_dataset

source, metadata = collect(seed=7, episodes=2, noise_levels=(0.,), random_fraction=0.5)
with TemporaryDirectory() as tmp:
    path = Path(tmp) / "sample"
    save_dataset(path, source, metadata)
    replay, record = load_dataset(path)
    for a, b in zip(source.episodes, replay.episodes, strict=True):
        for name in ("obs", "action", "reward", "terminated", "truncated", "dt"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    print([len(ep) for ep in replay.episodes], record["status_counts"])
PY
```

The offline run produced `[174, 345]` transitions, one crash and one safe landing, with 521 observation rows and 519 action rows. The same reader opened the local `s1-train` and legacy `m2r-train` collections (200 episodes and 59,362 transitions each, with and without `dt` respectively). Live KSP landing traces, frame calibration and event labeling remain pending.
