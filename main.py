"""Simulation entrypoint using the SEE planner."""

from __future__ import annotations
import sys
from typing import List, Optional
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
from simulation.planner import WaypointController, ExactWaypointController
from simulation.see_planner import SEEPlanner, SEEParameters, Viewpoint
from visualization.renderer import SimulationRenderer
from analysis.density_plot import plot_core_density


def build_default_environment() -> Environment:
    env = Environment(x_bounds=(-6.0, 6.0), y_bounds=(-4.5, 4.5), include_boundary_as_obstacles=False)
    # env.add_obstacle(LineSegmentObstacle((-5.5, 3.0), (5.5, 3.0)))
    # env.add_obstacle(LineSegmentObstacle((2.0, -3.5), (2.0, -1.0)))

    env.add_obstacle(
        FunctionObstacle(
            func=lambda x: 0.5 * np.sin(1.2 * x) + 3,
            x_min=-5.5,
            x_max=5.5,
            num_samples=100,
        )
    )

    def ellipse_param(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = np.array([2.5, 2.0])
        rx, ry = 1.4, 1.0
        return center[0] + rx * np.cos(u), center[1] + ry * np.sin(u)

    # env.add_obstacle(
    #     ParametricObstacle(
    #         param_func=ellipse_param,
    #         u_min=-0.2 * np.pi,
    #         u_max=0.9 * np.pi,
    #         num_samples=60,
    #     )
    # )
    return env


def run_simulation() -> None:

    DETAILED_ANIMATION: bool = False  # Step through classification with SPACE

    # 1. Instantiate environment and robot
    env = build_default_environment()
    robot_config = RobotConfig(
        radius=0.28,
        max_linear_velocity=1.4,
        max_angular_velocity=2.2,
    )
    start_state = RobotState(x=0.0, y=0.0, theta=np.pi/2)
    robot = Robot(state=start_state, config=robot_config)

    # 2. Configure sensor profile (Gamma_sensor)
    lidar_config = LidarConfig(
        num_rays=9,
        fov_rad=2.0 * np.pi /6.0,
        max_range=4.5,
        offset_angle=0.0,
        noise_std=0.005,
    )
    sensor = LidarSensor(config=lidar_config)

    # 3. Configure SEE algorithm parameters (Gamma_see)
    see_params = SEEParameters(
        epsilon= 0.008,       # Separation threshold to reject duplicate points
        # r_neighbor = 0.81,   # Neighborhood radius r
        r_neighbor = 0.6,
        k_min = 3,           # Minimum neighbors for core closure
        d_standoff = 3.0,    # d: Standoff distance along e_n for proposed views
        v_step = 0.12,       # v: Step size used in DirectNormal
        max_direct_iters = 15  # Loop cutoff for DirectNormal repeat-until
    )
    planner = SEEPlanner(gamma_see=see_params, gamma_sensor=lidar_config)
    controller = ExactWaypointController(position_tolerance=0.00, heading_tolerance=0.0, p_gain=1.5, rot_gain=2.5)

    # 4. Visualization and dwell timing
    renderer = SimulationRenderer(env, screen_size=(1300, 800))

    SCAN_PAUSE_DURATION = 0.6
    waypoint_dwell_timer = SCAN_PAUSE_DURATION

    # Initial scan at starting viewpoint
    sensor.scan(robot.state, env)
    visited_waypoints: List[np.ndarray] = [robot.state.position.copy()]
    
    # Initial planner step
    current_target: Optional[Viewpoint] = None
    if DETAILED_ANIMATION:
        planner.start_detailed_classification(robot.state, sensor.last_scan)
    else:
        current_target = planner.select_next_waypoint(robot, env, sensor.last_scan)

    # 5. Timing loop variables
    DT = 0.02
    accumulator = 0.0
    sim_time = 0.0
    is_paused = False

    trajectory = [(robot.state.x, robot.state.y)]
    last_clock_time = pygame.time.get_ticks() / 1000.0

    running = True
    while running:
        current_clock_time = pygame.time.get_ticks() / 1000.0
        frame_time = current_clock_time - last_clock_time
        last_clock_time = current_clock_time

        if frame_time > 0.2:
            frame_time = 0.2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if DETAILED_ANIMATION and planner.active_classifier_gen is not None:
                        # Advance one classification step in Algorithm 3
                        has_more = planner.step_detailed_classification()
                        if not has_more:
                            v_c = Viewpoint(position=robot.state.position.copy(), heading=robot.state.theta)
                            planner.propose_views(v_c)
                            next_view = planner.select_nbv(v_c)
                            if next_view is not None:
                                current_target = next_view
                    else:
                        is_paused = not is_paused

                elif event.key == pygame.K_d:
                    # Toggle detailed animation mode
                    DETAILED_ANIMATION = not DETAILED_ANIMATION
                    if not DETAILED_ANIMATION and planner.active_classifier_gen is not None:
                        while planner.step_detailed_classification():
                            pass
                        v_c = Viewpoint(position=robot.state.position.copy(), heading=robot.state.theta)
                        planner.propose_views(v_c)
                        if next_view is not None:
                            current_target = next_view

                elif event.key == pygame.K_r:
                    robot.state = RobotState(x=-4.5, y=-3.0, theta=0.0)
                    planner = SEEPlanner(gamma_see=see_params, gamma_sensor=lidar_config)
                    sensor.clear_accumulated_hits()
                    visited_waypoints = [robot.state.position.copy()]
                    trajectory = [(robot.state.x, robot.state.y)]
                    sim_time = 0.0
                    sensor.scan(robot.state, env)
                    wp_index = 0
                    current_target = candidate_waypoints[wp_index]
                    waypoint_dwell_timer = SCAN_PAUSE_DURATION
                    if DETAILED_ANIMATION:
                        planner.start_detailed_classification(robot.state, sensor.last_scan)
                    else:
                        planner.classify_points(Viewpoint(position=robot.state.position.copy(), heading=robot.state.theta))

                elif event.key == pygame.K_l:
                    renderer.show_rays = not renderer.show_rays
                elif event.key == pygame.K_o:
                    renderer.show_accumulated = not renderer.show_accumulated
                elif event.key == pygame.K_t:
                    renderer.show_trajectory = not renderer.show_trajectory

        # Normal physics update loop (held stationary while detailed classification is active)
        is_waiting_for_user_steps = DETAILED_ANIMATION and (planner.active_classifier_gen is not None)

        if not is_paused and not is_waiting_for_user_steps:
            accumulator += frame_time
            while accumulator >= DT:
                sensor.update(DT)

                if waypoint_dwell_timer > 0.0:
                    waypoint_dwell_timer = max(0.0, waypoint_dwell_timer - DT)
                    vx, vy, omega = 0.0, 0.0, 0.0

                    if waypoint_dwell_timer == 0.0:
                        current_target = planner.select_next_waypoint(robot, env, sensor.last_scan)
                else:
                    if current_target is not None and controller.has_arrived(
                        robot.state.position,
                        current_target.position,
                        robot.state.theta,
                        current_target.heading,
                    ):
                        vx, vy, omega = 0.0, 0.0, 0.0
                        sensor.scan(robot.state, env)
                        visited_waypoints.append(robot.state.position.copy())
                        waypoint_dwell_timer = SCAN_PAUSE_DURATION

                        if DETAILED_ANIMATION:
                            planner.start_detailed_classification(robot.state, sensor.last_scan)
                        else:
                            v_c = Viewpoint(position=robot.state.position.copy(), heading=robot.state.theta)
                            planner.get_new_measurements(v_c, sensor.last_scan)
                            planner.classify_points(v_c)
                            planner.propose_views(v_c)
                    elif current_target is not None:
                        vx, vy, omega = controller.compute_command(
                            robot_state=robot.state,
                            target_pos=current_target.position,
                            target_heading=current_target.heading,
                            dt=DT,
                            max_v=robot.config.max_linear_velocity,
                            max_omega=robot.config.max_angular_velocity,
                        )
                    else:
                        vx, vy, omega = 0.0, 0.0, 0.0

                robot.set_control(vx, vy, omega)
                robot.integrate(DT, env)

                trajectory.append((robot.state.x, robot.state.y))
                if len(trajectory) > 2000:
                    trajectory.pop(0)

                sim_time += DT
                accumulator -= DT

        # Safely resolve coordinates for renderer
        if current_target is not None:
            target_coords = current_target.position if hasattr(current_target, "position") else current_target
        else:
            target_coords = None

        renderer.render(
            env=env,
            robot=robot,
            sensor=sensor,
            current_target=current_target,
            visited_waypoints=visited_waypoints,
            trajectory=trajectory,
            sim_time=sim_time,
            is_paused=is_paused,
            planner=planner,
            detailed_animation=DETAILED_ANIMATION,
        )

        renderer.clock.tick(60)

    # Exit the simulation loop
    pygame.quit()

    # Generate and display density plot using planner's neighborhood radius r
    print(f"Total Core Points (P_c): {len(planner.P_c)}")
    plot_core_density(
        core_points=planner.P_c,
        x_bounds=(env.x_min, env.x_max),
        sample_step=0.05,
        r_neighbor=planner.gamma_see.r_neighbor,
    )

    sys.exit(0)


if __name__ == "__main__":
    run_simulation()