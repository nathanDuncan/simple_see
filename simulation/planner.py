"""Waypoint controllers and decoupled planner interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from .environment import Environment
from .robot import Robot, RobotState
from .sensor import RayHit


class WaypointController:
    """
    Proportional holonomic controller for tracking 2D waypoints.
    Produces translational (vx, vy) commands and aligns heading with travel.
    """

    def __init__(self, position_tolerance: float = 0.15, p_gain: float = 1.8, rot_gain: float = 3.0):
        self.position_tolerance = position_tolerance
        self.p_gain = p_gain
        self.rot_gain = rot_gain

    def has_arrived(self, robot_pos: np.ndarray, target: np.ndarray) -> bool:
        return float(np.linalg.norm(target - robot_pos)) <= self.position_tolerance

    def compute_command(
        self, robot_state: RobotState, target: np.ndarray
    ) -> tuple[float, float, float]:
        """
        Compute control command (vx, vy, omega).
        Handles zero-distance edge cases gracefully.
        """
        current_pos = robot_state.position
        error = target - current_pos
        dist = np.linalg.norm(error)

        if dist < 1e-4:
            return 0.0, 0.0, 0.0

        # Translational control
        v_cmd = self.p_gain * error

        # Heading alignment towards motion direction
        desired_heading = np.arctan2(error[1], error[0])
        heading_error = (desired_heading - robot_state.theta + np.pi) % (2.0 * np.pi) - np.pi
        omega_cmd = self.rot_gain * heading_error

        return float(v_cmd[0]), float(v_cmd[1]), float(omega_cmd)


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