"""Waypoint controllers and decoupled planner interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from .environment import Environment
from .robot import Robot, RobotState
from .sensor import RayHit


class WaypointController:
    def __init__(
        self,
        position_tolerance: float = 0.22,
        heading_tolerance: float = 0.05,  # ~2.8 degrees
        p_gain: float = 1.6,
        rot_gain: float = 3.5,
    ):
        self.position_tolerance = position_tolerance
        self.heading_tolerance = heading_tolerance
        self.p_gain = p_gain
        self.rot_gain = rot_gain

    def has_arrived(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
        robot_theta: Optional[float] = None,
        target_theta: Optional[float] = None,
    ) -> bool:
        pos_dist = float(np.linalg.norm(target_pos - robot_pos))
        if pos_dist > self.position_tolerance:
            return False

        if target_theta is not None and robot_theta is not None:
            heading_err = (target_theta - robot_theta + np.pi) % (2.0 * np.pi) - np.pi
            return abs(heading_err) <= self.heading_tolerance

        return True

    def compute_command(
        self,
        robot_state: RobotState,
        target_pos: np.ndarray,
        target_heading: Optional[float] = None,
    ) -> tuple[float, float, float]:
        current_pos = robot_state.position
        error = target_pos - current_pos
        dist = float(np.linalg.norm(error))

        if dist > self.position_tolerance:
            # Transit phase: translate toward target while looking ahead along motion path
            v_cmd = self.p_gain * error
            desired_heading = np.arctan2(error[1], error[0])
        else:
            # Arrived at position: stop translation and turn in-place to target_heading
            v_cmd = np.zeros(2)
            desired_heading = target_heading if target_heading is not None else robot_state.theta

        heading_error = (desired_heading - robot_state.theta + np.pi) % (2.0 * np.pi) - np.pi

        # Apply minimum angular speed if not within deadband to prevent stall
        if abs(heading_error) <= self.heading_tolerance:
            omega_cmd = 0.0
        else:
            omega_cmd = np.clip(self.rot_gain * heading_error, -2.2, 2.2)
            # Avoid vanishingly small velocities that fail to move against timestep
            if abs(omega_cmd) < 0.2:
                omega_cmd = 0.2 * np.sign(heading_error)

        return float(v_cmd[0]), float(v_cmd[1]), float(omega_cmd)

class ExactWaypointController:
    """
    Finite-time deadbeat holonomic controller.
    Drives position and orientation to exact values in finite discrete steps.
    """

    def __init__(
        self,
        position_tolerance: float = 0.0,
        heading_tolerance: float = 0.0,
        p_gain: float = 2.0,
        rot_gain: float = 4.0,
    ):
        self.position_tolerance = position_tolerance
        self.heading_tolerance = heading_tolerance
        self.p_gain = p_gain
        self.rot_gain = rot_gain

    def has_arrived(
        self,
        robot_pos: np.ndarray,
        target_pos: np.ndarray,
        robot_theta: Optional[float] = None,
        target_theta: Optional[float] = None,
    ) -> bool:
        """
        Evaluates exact arrival using tolerances (supports 0.0 with float precision guard).
        """
        pos_dist = float(np.linalg.norm(target_pos - robot_pos))
        effective_pos_tol = max(self.position_tolerance, 1e-6)
        if pos_dist > effective_pos_tol:
            return False

        if target_theta is not None and robot_theta is not None:
            heading_err = (target_theta - robot_theta + np.pi) % (2.0 * np.pi) - np.pi
            effective_head_tol = max(self.heading_tolerance, 1e-6)
            return abs(heading_err) <= effective_head_tol

        return True

    def compute_command(
        self,
        robot_state: RobotState,
        target_pos: np.ndarray,
        target_heading: Optional[float] = None,
        dt: float = 0.02,
        max_v: float = 1.4,
        max_omega: float = 2.2,
    ) -> tuple[float, float, float]:
        """
        Computes deadbeat velocities that land exactly on target in finite time.
        """
        current_pos = robot_state.position
        error = target_pos - current_pos
        dist = float(np.linalg.norm(error))

        # 1. Exact Linear Velocity
        # The velocity required to close the entire remaining distance in one timestep:
        needed_v = error / dt
        needed_speed = float(np.linalg.norm(needed_v))

        if needed_speed <= max_v:
            # Reached in 1 step: command exact deadbeat velocity
            vx, vy = float(needed_v[0]), float(needed_v[1])
        else:
            # Saturated by max speed: move along direct vector at max speed
            scale = max_v / needed_speed
            vx, vy = float(needed_v[0] * scale), float(needed_v[1] * scale)

        # 2. Exact Angular Velocity
        if target_heading is not None:
            # If still far away, face the direction of motion; once close, face target heading
            if dist > 0.3:
                desired_heading = np.arctan2(error[1], error[0])
            else:
                desired_heading = target_heading
        else:
            if dist > 1e-4:
                desired_heading = np.arctan2(error[1], error[0])
            else:
                desired_heading = robot_state.theta

        heading_error = (desired_heading - robot_state.theta + np.pi) % (2.0 * np.pi) - np.pi

        # The angular rate required to close heading error in one timestep:
        needed_omega = heading_error / dt

        if abs(needed_omega) <= max_omega:
            # Reached in 1 step: command exact deadbeat rotational rate
            omega = float(needed_omega)
        else:
            omega = float(np.sign(needed_omega) * max_omega)

        return vx, vy, omega

class BasePlanner(ABC):
    """
    Abstract exploration planner interface.
    Decoupled from sensor and controller to support future algorithms
    (e.g., frontier exploration, RRT*, PRM, or mutual information gain).
    """

    @abstractmethod
    def select_next_waypoint(
        self,
        robot: Robot,
        environment: Environment,
        observations: List[RayHit],
    ) -> Optional[np.ndarray]:
        """Select next target waypoint [x, y]."""
        pass


class CyclicWaypointPlanner(BasePlanner):
    """
    Deterministic placeholder planner cycling through candidate waypoints.
    
    LIMITATION NOTE:
    This placeholder does not perform obstacle-free path validation or dynamic
    replanning. It directly issues waypoints. Advanced mapping and collision avoidance
    (such as visibility graphs, bug algorithms, or A*) can drop directly into this interface.
    """

    def __init__(self, waypoints: List[tuple[float, float]]):
        if not waypoints:
            raise ValueError("Waypoints list cannot be empty.")
        self.waypoints = [np.array(wp, dtype=np.float64) for wp in waypoints]
        self.current_idx = 0
        self.visited_waypoints: List[np.ndarray] = []

    @property
    def current_target(self) -> np.ndarray:
        return self.waypoints[self.current_idx]

    def advance_to_next(self) -> np.ndarray:
        self.visited_waypoints.append(self.current_target.copy())
        self.current_idx = (self.current_idx + 1) % len(self.waypoints)
        return self.current_target

    def select_next_waypoint(
        self,
        robot: Robot,
        environment: Environment,
        observations: List[RayHit],
    ) -> Optional[np.ndarray]:
        return self.current_target