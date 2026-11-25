# Nominal Mun descent decisions

On 2026-11-25 the project lead approved the S3 task values for nominal KSP descent. The KSP integration group will identify a flat site, supply its measured coordinates and clearance reference, and provide a reproducible descent save. The team will choose how to reconcile the craft and simulator **after measuring the craft**. The orbit test save and its sub-satellite coordinates identify the probe used to collect existing evidence. The authoritative fields are in the [shared task configuration](../configs/mun_landing.toml).

## Approved task values

| Quantity | Decision |
| --- | --- |
| Initial position | Draw each local horizontal coordinate independently and uniformly from −30 to +30 m. Draw vertical clearance uniformly from 80 to 120 m. |
| Initial velocity | In the target-relative, upward-positive convention, draw each horizontal component independently and uniformly from −2 to +2 m/s and vertical velocity from −8 to −3 m/s. |
| Initial attitude and rate | Draw the thrust direction uniformly in solid angle in a 12° cone about vertical, with uniform azimuth and yaw over a full turn. Draw body angular velocity uniformly by volume inside a ball of radius 0.05 rad/s. |
| Control interval and horizon | Hold each action for 0.1 s of game time (10 Hz). Time out at 90 s. Record the actual UT interval on every transition. |
| Touchdown and stability | Total speed ≤2.0 m/s, tilt ≤10°, angular-speed norm ≤0.2 rad/s (11.4591559026°/s). Require 5 consecutive seconds of stable contact and an intact vessel. |
| Precision | Target radius 5.0 m. Report precision across all attempts and safe-in-target separately. |
| Observation | Retain position, velocity, orientation, body angular velocity and main-engine propellant. Record mass and inertia for diagnostics only. Source checkpoint preprocessing is recorded in the handoff. |

The source reset sampler starts with a full main tank. The KSP integration group must supply the applicable propellant resource, units and usable mass for the selected configuration. The KSP contact signal, terrain reference and repeatable reset still need live acceptance before any readiness flag changes. The exact source sampling code and the nominal pilot configuration are available for comparison. [Reset sampler](../simulator/src/mun_lander/state.py) · [Pilot configuration](../simulator/configs/nominal.json)

## Decision deferred until KSP measurements

The recorded 5% throttle pulse peaked at 12 kN. A linear extrapolation gives about 240 kN at full throttle, versus 9 kN in the pilot simulator. The short RCS pulses suggest about 0.07 rad/s² at full input on the tested configuration. The pilot's nominal torque-to-inertia ratios are 0.75, 0.75 and 0.50 rad/s² by axis. Compare signs and gains per axis after the frames are mapped. The observed throttle lag also matters on a 0.1 s control interval. [Game measurements](validation.md) · [Adapter test configuration](../configs/ksp_mun_test.toml)

| Owner | Evidence to return | Decision it enables |
| --- | --- | --- |
| KSP integration | Identify the exact craft, stage, enabled engines, propellant level, craft/save hashes and date. For this configuration record wet/dry mass, principal inertias and axes, thrust curve and delay, Isp, three-axis RCS response, and resource-to-kg conversion. Supply timestamped measurements with units. | Compare acceleration and controllability at matched fuel loads in the two domains. |
| KSP integration | Identify the site and named descent snapshot. Supply target latitude, longitude, altitude/reference body, the clearance reference, and readback after repeated resets. | Populate the destination and save fields and verify the approved initial-state envelope is reproducible. |
| Simulator and world-model teams | Compare thrust divided by mass, estimated hover throttle at local gravity, and torque divided by principal inertia for each axis. Account for measured delay. | Present the measured cost of keeping the current craft versus limiting main thrust/changing RCS on the craft. The team then records the chosen route before physical recalibration and retraining. |

For context, the pilot wet mass is 1,800 kg. At 1.63 m/s² nominal gravity and 9 kN maximum thrust, its wet hover throttle is approximately 0.326. The corresponding KSP estimate requires wet mass and the actual thrust curve. The task-level thresholds above can be used immediately for offline interface work.
