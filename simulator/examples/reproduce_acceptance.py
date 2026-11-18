"""Reproduce saved simulator acceptance artifacts with no learned-policy training.

Run after `pip install -e '.[demo]'`: python examples/reproduce_acceptance.py
The test seed and generalization seed namespaces are disjoint. The engineering
controller and nominal parameters are held fixed throughout the sweep.
"""

from pathlib import Path

from mun_lander import SimConfig
from mun_lander.cli import _write_json, save_trajectory
from mun_lander.evaluation import (
    benchmark, evaluate, generate_initial_states, parameter_sweep, rollout,
)


def main():
    output = Path("artifacts")
    config = SimConfig()
    test_starts = generate_initial_states(config, 100, 2026, "test")
    _write_json(output / "test_starts.json", test_starts)
    evaluation = evaluate(config, test_starts)
    _write_json(output / "evaluation_scripted.json", evaluation)
    print("Nominal scripted:", evaluation["status_counts"], flush=True)
    small_starts = generate_initial_states(config, 10, 2026, "test")
    _write_json(output / "smoke_starts.json", small_starts)
    for controller in ("zero", "random"):
        evaluation = evaluate(config, small_starts, controller)
        _write_json(output / f"evaluation_{controller}.json", evaluation)
        print(controller, evaluation["status_counts"], flush=True)
    generalization_starts = generate_initial_states(config, 10, 2026, "generalization")
    _write_json(output / "generalization_starts.json", generalization_starts)
    sweep = parameter_sweep(config, generalization_starts, scales=(0.8, 1.0, 1.2))
    _write_json(output / "sweep.json", sweep)
    print("Parameter sweep:", [(r["parameter"], r["scale"], r["evaluation"]["status_counts"])
                               for r in sweep["runs"]], flush=True)
    result, trajectory = rollout(config, seed=7, record=True)
    save_trajectory(output / "demo", trajectory, result, config)
    measurement = benchmark(config, steps=2000, seed=7)
    _write_json(output / "benchmark.json", measurement)
    print("Benchmark:", round(measurement["transitions_per_second"]), "transitions/s", flush=True)


if __name__ == "__main__":
    main()
