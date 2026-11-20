import json

import pytest

from mun_lander import SimConfig


def test_config_json_roundtrip(tmp_path):
    config = SimConfig()
    path = tmp_path / "nominal.json"
    config.to_json(path)
    restored = SimConfig.from_json(path)
    assert restored == config
    assert SimConfig.from_dict(json.loads(json.dumps(config.to_dict()))).to_dict() == config.to_dict()


@pytest.mark.parametrize("section,key,value", [
    ("vehicle", "gravity", -1), ("vehicle", "dry_mass", 0),
    ("vehicle", "fuel_capacity", -1), ("vehicle", "inertia", [1, 0, 1]),
    ("vehicle", "main_isp", float("nan")), ("landing", "stability_time", -1),
    (None, "control_dt", 0), (None, "physics_dt", -0.1),
    (None, "max_time", float("inf")), ("initial", "altitude", [120, 80]),
])
def test_invalid_configuration_rejected(section, key, value):
    data = SimConfig().to_dict()
    destination = data if section is None else data[section]
    destination[key] = value
    with pytest.raises((ValueError, TypeError)):
        SimConfig.from_dict(data)


def test_independent_generalization_parameters():
    nominal = SimConfig()
    v = nominal.vehicle
    mass = nominal.perturbed("mass", 1.1).vehicle
    assert mass.dry_mass + mass.fuel_capacity == pytest.approx(1.1 * (v.dry_mass + v.fuel_capacity))
    assert mass.fuel_capacity == v.fuel_capacity
    assert mass.inertia == v.inertia and mass.max_thrust == v.max_thrust and mass.gravity == v.gravity
    for parameter, attribute in [("inertia", "inertia"), ("thrust", "max_thrust"), ("gravity", "gravity")]:
        altered = nominal.perturbed(parameter, 1.1).vehicle
        original = v.__dict__
        modified = altered.__dict__
        for key in original:
            if key != attribute:
                assert modified[key] == original[key]
    assert nominal.perturbed("gravity", 1.0) == nominal
    with pytest.raises(ValueError):
        nominal.perturbed("gravity", -1)
    with pytest.raises(ValueError):
        nominal.perturbed("unknown", 1.1)


def test_initial_horizontal_box_fits_circular_simulation_domain():
    data = SimConfig().to_dict()
    # Each coordinate can reach 30 m, so box corners exceed a 40 m radius.
    data["max_horizontal_distance"] = 40
    with pytest.raises(ValueError):
        SimConfig.from_dict(data)
