"""Physical reference checks, independent of a trained or scripted controller."""
from dataclasses import replace

import numpy as np
import pytest

from mun_lander.config import VehicleConfig
from mun_lander.math3d import (
    normalize_quaternion,
    quat_multiply,
    quaternion_from_euler,
    rotation_matrix,
    tilt_angle,
)
from mun_lander.physics import integrate_free
from mun_lander.state import LanderState


def state(**updates):
    fields = dict(
        position=np.array([0.0, 0.0, 100.0]),
        velocity=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        fuel=600.0,
    )
    fields.update(updates)
    return LanderState(**fields)


def test_free_fall_matches_analytic_solution():
    vehicle = VehicleConfig()
    initial = state(velocity=np.array([2.0, -3.0, 4.0]))
    result, rcs = integrate_free(initial, np.zeros(4), vehicle, 0.25)
    acceleration = np.array([0.0, 0.0, -vehicle.gravity])
    np.testing.assert_allclose(
        result.position,
        initial.position + initial.velocity * 0.25 + 0.5 * acceleration * 0.25**2,
        atol=1e-10,
    )
    np.testing.assert_allclose(result.velocity, initial.velocity + acceleration * 0.25, atol=1e-10)
    assert result.fuel == initial.fuel
    assert rcs == 0
    np.testing.assert_array_equal(initial.position, [0.0, 0.0, 100.0])


def test_variable_mass_thrust_matches_rocket_equation():
    vehicle = VehicleConfig()
    initial = state()
    dt = 1.0
    result, rcs = integrate_free(initial, np.array([1.0, 0.0, 0.0, 0.0]), vehicle, dt)
    exhaust_speed = vehicle.main_isp * 9.80665
    expected_used = vehicle.max_thrust / exhaust_speed * dt
    expected_vz = exhaust_speed * np.log(
        (vehicle.dry_mass + initial.fuel) / (vehicle.dry_mass + initial.fuel - expected_used)
    ) - vehicle.gravity * dt
    assert result.fuel == pytest.approx(initial.fuel - expected_used, abs=1e-10)
    assert result.velocity[2] == pytest.approx(expected_vz, abs=1e-8)
    np.testing.assert_allclose(result.velocity[:2], 0, atol=1e-12)
    assert rcs == 0


def test_burnout_has_no_negative_fuel_or_extra_impulse():
    vehicle = VehicleConfig()
    initial = state(fuel=0.01)
    dt = 0.5
    result, _ = integrate_free(initial, np.array([1.0, 0.0, 0.0, 0.0]), vehicle, dt)
    exhaust_speed = vehicle.main_isp * 9.80665
    expected_vz = exhaust_speed * np.log((vehicle.dry_mass + initial.fuel) / vehicle.dry_mass) - vehicle.gravity * dt
    assert result.fuel == pytest.approx(0, abs=1e-12)
    assert result.velocity[2] == pytest.approx(expected_vz, abs=1e-8)


def test_thrust_rotates_from_body_to_world():
    vehicle = VehicleConfig()
    # +90 degree pitch maps the body's +z thrust axis to world +x.
    initial = state(quaternion=quaternion_from_euler(0.0, np.pi / 2, 0.0))
    result, _ = integrate_free(initial, np.array([1.0, 0.0, 0.0, 0.0]), vehicle, 0.01)
    assert result.velocity[0] > 0
    assert result.velocity[1] == pytest.approx(0, abs=1e-12)
    assert result.velocity[2] == pytest.approx(-vehicle.gravity * 0.01, abs=1e-10)


def test_principal_axis_torque_and_rcs_accounting():
    vehicle = VehicleConfig()
    dt = 0.2
    result, rcs = integrate_free(state(), np.array([0.0, 1.0, 0.0, 0.0]), vehicle, dt)
    expected_omega = vehicle.rcs_torque[0] / vehicle.inertia[0] * dt
    np.testing.assert_allclose(result.angular_velocity, [expected_omega, 0.0, 0.0], atol=1e-10)
    assert result.quaternion[1] > 0
    assert result.fuel == 600.0
    assert rcs > 0
    _, half_rcs = integrate_free(state(), np.array([0.0, 0.5, 0.0, 0.0]), vehicle, dt)
    assert half_rcs == pytest.approx(rcs / 2, rel=1e-10)


def test_torque_free_energy_and_world_angular_momentum():
    vehicle = replace(VehicleConfig(), inertia=(900.0, 1400.0, 2000.0))
    current = state(angular_velocity=np.array([0.3, -0.2, 0.4]))
    inertia = np.asarray(vehicle.inertia)
    initial_energy = 0.5 * np.dot(inertia * current.angular_velocity, current.angular_velocity)
    initial_momentum = rotation_matrix(current.quaternion) @ (inertia * current.angular_velocity)
    for _ in range(200):
        current, _ = integrate_free(current, np.zeros(4), vehicle, 0.01)
    energy = 0.5 * np.dot(inertia * current.angular_velocity, current.angular_velocity)
    momentum = rotation_matrix(current.quaternion) @ (inertia * current.angular_velocity)
    assert energy == pytest.approx(initial_energy, rel=1e-8)
    np.testing.assert_allclose(momentum, initial_momentum, rtol=1e-8, atol=1e-6)
    assert np.linalg.norm(current.quaternion) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_conventions_and_rotation_invariants():
    q = quaternion_from_euler(0.2, -0.3, 0.4)
    rotation = rotation_matrix(q)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(rotation_matrix(-q), rotation, atol=1e-12)
    inverse = q * np.array([1.0, -1.0, -1.0, -1.0])
    np.testing.assert_allclose(quat_multiply(q, inverse), [1.0, 0.0, 0.0, 0.0], atol=1e-12)
    assert tilt_angle(quaternion_from_euler(0.0, 0.0, 1.7)) == pytest.approx(0.0, abs=1e-12)
    assert tilt_angle(quaternion_from_euler(0.3, 0.0, 0.0)) == pytest.approx(0.3, abs=1e-12)
    assert np.linalg.norm(normalize_quaternion(q * 7.0)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        normalize_quaternion(np.zeros(4))
