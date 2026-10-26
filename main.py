"""Simulation entrypoint and decoupled fixed-step accumulator execution loop."""

from __future__ import annotations
import sys
import numpy as np
import pygame

from simulation.environment import (
    Environment,
    LineSegmentObstacle,
    FunctionObstacle,
    ParametricObstacle,
)
from simulation.robot import Robot, RobotState, RobotConfig
from simulation.sensor import LidarSensor, LidarConfig
from simulation.planner import WaypointController, CyclicWaypointPlanner
from visualization.renderer import SimulationRenderer


def build_default_environment() -> Environment:
    """Instantiate workspace with lines, function curves, and parametric curves."""
    env = Environment(x_bounds=(-6.0, 6.0), y_bounds=(-4.5, 4.5), include_boundary_as_obstacles=True)

    # 1. Finite line segments (interior walls)
    env.add_obstacle(LineSegmentObstacle((-3.5, 2.5), (-1.5, 2.5)))
    env.add_obstacle(LineSegmentObstacle((2.0, -3.5), (2.0, -1.0)))

    # 2. Continuous function-based curve: y = sin(x) + offset
    env.add_obstacle(
        FunctionObstacle(
            func=lambda x: 0.8 * np.sin(1.2 * x) - 1.5,
            x_min=-4.5,
            x_max=0.5,
            num_samples=75,
        )
    )

    # 3. Continuous parametric curve: Elliptic arc obstacle
    def ellipse_param(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.array([2.5, 2.0])
        rx, ry = 1.4, 1.0
        return center[0] + rx * np.cos(u), center[1] + ry * np.sin(u)

    env.add_obstacle(
        ParametricObstacle(
            param_func=ellipse_param,
            u_min=-0.2 * np.pi,
            u_max=0.9 * np.pi,
            num_samples=60,
        )
    )

    return env


def run_simulation() -> None:
    # 1. Setup simulation components
    env = build_default_environment()
    robot_config = RobotConfig(radius=0.28, max_linear_velocity=1.4, max_angular_velocity=2.2)
    robot = Robot(state=RobotState(x=-4.5, y=-3.0, theta=0.0), config=robot_config)

    lidar_config = LidarConfig(
        num_rays=150,
        fov_rad=2.0 * np.pi,
        max_range=4.5,
        offset_angle=0.0,
        noise_std=0.005,
        sweep_rate_hz=2.0,
    )
    sensor = LidarSensor(config=lidar_config)

    waypoints = [
        (-4.5, -3.0),
        (-3.0, 0.5),
        (-0.5, 3.2),
        (3.5, 3.0),
        (4.2, -0.5),
        (0.0, 0.0),
        (-2.0, -2.8),
    ]
    planner = CyclicWaypointPlanner(waypoints=waypoints)
    controller = WaypointController(position_tolerance=0.20, p_gain=1.5, rot_gain=2.5)

    renderer = SimulationRenderer(env, screen_size=(1024, 768))

    # 2. Timing and accumulator setup
    DT = 0.02  # Fixed simulation physics step (50 Hz)
    accumulator = 0.0
    sim_time = 0.0
    is_paused = False

    trajectory = [(robot.state.x, robot.state.y)]
    last_clock_time = pygame.time.get_ticks() / 1000.0

    running = True
    while running:
        # A. Frame timing calculation
        current_clock_time = pygame.time.get_ticks() / 1000.0
        frame_time = current_clock_time - last_clock_time
        last_clock_time = current_clock_time

        # Cap frame time to prevent accumulator spiral after window freeze or drag
        if frame_time > 0.2:
            frame_time = 0.2

        # B. Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    is_paused = not is_paused
                elif event.key == pygame.K_r:
                    # Reset
                    robot.state = RobotState(x=-4.5, y=-3.0, theta=0.0)
                    planner.current_idx = 0
                    planner.visited_waypoints.clear()
                    sensor.clear_accumulated_hits()
                    trajectory.clear()
                    trajectory.append((robot.state.x, robot.state.y))
                    sim_time = 0.0
                elif event.key == pygame.K_l:
                    renderer.show_rays = not renderer.show_rays
                elif event.key == pygame.K_o:
                    renderer.show_accumulated = not renderer.show_accumulated
                elif event.key == pygame.K_t:
                    renderer.show_trajectory = not renderer.show_trajectory

        # C. Fixed-timestep simulation update loop
        if not is_paused:
            accumulator += frame_time
            while accumulator >= DT:
                # 1. Waypoint Selection & Arrival check
                target_wp = planner.select_next_waypoint(robot, env, sensor.last_scan)
                if target_wp is not None:
                    if controller.has_arrived(robot.state.position, target_wp):
                        target_wp = planner.advance_to_next()

                # 2. Motion Control
                if target_wp is not None:
                    vx, vy, omega = controller.compute_command(robot.state, target_wp)
                else:
                    vx, vy, omega = 0.0, 0.0, 0.0
                robot.set_control(vx, vy, omega)

                # 3. Kinematic Integration & Boundary Clamping
                robot.integrate(DT, env)

                # 4. Sensor Sweep & Observation Update
                sensor.update_sweep(robot.state, DT, env)

                # 5. History Recording
                trajectory.append((robot.state.x, robot.state.y))
                if len(trajectory) > 2000:
                    trajectory.pop(0)

                sim_time += DT
                accumulator -= DT

        # D. Rendering (Decoupled from simulation physics rate)
        renderer.render(
            env=env,
            robot=robot,
            sensor=sensor,
            current_target=planner.current_target,
            visited_waypoints=planner.visited_waypoints,
            trajectory=trajectory,
            sim_time=sim_time,
            is_paused=is_paused,
        )

        renderer.clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    run_simulation()