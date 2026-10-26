"""Unit tests verifying angle wrapping, kinematics, boundary limits, and ray-casting."""

import numpy as np
import pytest

from simulation.environment import (
    wrap_angle,
    Environment,
    LineSegmentObstacle,
    FunctionObstacle,
)
from simulation.robot import Robot, RobotState, RobotConfig
from simulation.sensor import LidarSensor, LidarConfig
from simulation.planner import WaypointController


def test_angle_wrapping():
    """Verify wrap_angle stays within [-pi, pi)."""
    assert np.isclose(wrap_angle(0.0), 0.0)
    assert np.isclose(wrap_angle(np.pi), -np.pi)
    assert np.isclose(wrap_angle(3.0 * np.pi), -np.pi)
    assert np.isclose(wrap_angle(-3.0 * np.pi), -np.pi)
    assert np.isclose(wrap_angle(np.pi / 2.0), np.pi / 2.0)
    assert np.isclose(wrap_angle(2.5 * np.pi), 0.5 * np.pi)


def test_robot_kinematic_integration():
    """Verify velocity-based displacement and orientation updates."""
    env = Environment(x_bounds=(-10.0, 10.0), y_bounds=(-10.0, 10.0), include_boundary_as_obstacles=False)
    robot = Robot(state=RobotState(x=0.0, y=0.0, theta=0.0))
    robot.set_control(vx=1.0, vy=-0.5, omega=np.pi)
    dt = 0.1
    robot.integrate(dt, env)

    assert np.isclose(robot.state.x, 0.1)
    assert np.isclose(robot.state.y, -0.05)
    assert np.isclose(robot.state.theta, 0.1 * np.pi)


def test_workspace_boundary_enforcement():
    """Verify robot cannot translate beyond workspace bounds minus its radius."""
    env = Environment(x_bounds=(-2.0, 2.0), y_bounds=(-2.0, 2.0), include_boundary_as_obstacles=False)
    cfg = RobotConfig(radius=0.5, max_linear_velocity=10.0)
    robot = Robot(state=RobotState(x=1.4, y=0.0, theta=0.0), config=cfg)

    # Command translation that would penetrate the boundary x = 2.0
    robot.set_control(vx=2.0, vy=0.0, omega=0.0)
    robot.integrate(dt=1.0, env=env)

    # Max permissible x is x_max - radius = 2.0 - 0.5 = 1.5
    assert np.isclose(robot.state.x, 1.5)
    assert robot.state.vx == 0.0


def test_ray_segment_intersection():
    """Verify ray casting hits an obstacle at the exact Euclidean distance."""
    env = Environment(x_bounds=(-10.0, 10.0), y_bounds=(-10.0, 10.0), include_boundary_as_obstacles=False)
    # Vertical wall at x = 3.0 from y = -2.0 to y = 2.0
    env.add_obstacle(LineSegmentObstacle((3.0, -2.0), (3.0, 2.0)))

    cfg = LidarConfig(num_rays=1, fov_rad=0.0, max_range=5.0, noise_std=0.0)
    sensor = LidarSensor(config=cfg)

    # Robot at (0.0, 0.0) pointing directly right along +X
    robot_state = RobotState(x=0.0, y=0.0, theta=0.0)
    hits = sensor.cast_all_rays(robot_state, env)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.detected is True
    assert np.isclose(hit.range, 3.0)
    assert np.allclose(hit.hit_point, [3.0, 0.0])


def test_ray_miss_and_range_cutoff():
    """Verify rays passing by obstacles or beyond max range report detected=False."""
    env = Environment(x_bounds=(-10.0, 10.0), y_bounds=(-10.0, 10.0), include_boundary_as_obstacles=False)
    # Wall located at x = 8.0
    env.add_obstacle(LineSegmentObstacle((8.0, -2.0), (8.0, 2.0)))

    # Sensor with max range 4.0 meters
    cfg = LidarConfig(num_rays=1, fov_rad=0.0, max_range=4.0, noise_std=0.0)
    sensor = LidarSensor(config=cfg)

    robot_state = RobotState(x=0.0, y=0.0, theta=0.0)
    hits = sensor.cast_all_rays(robot_state, env)

    assert len(hits) == 1
    hit = hits[0]
    assert hit.detected is False
    assert hit.hit_point is None
    assert np.isclose(hit.range, 4.0)


def test_controller_zero_distance():
    """Verify waypoint controller gracefully handles arrival and near-zero distance."""
    controller = WaypointController(position_tolerance=0.1)
    state = RobotState(x=2.0, y=3.0, theta=0.0)
    target = np.array([2.0, 3.0])

    assert controller.has_arrived(state.position, target) is True
    vx, vy, omega = controller.compute_command(state, target)
    assert vx == 0.0 and vy == 0.0 and omega == 0.0