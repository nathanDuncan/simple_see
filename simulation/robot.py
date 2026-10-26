"""Holonomic 2D kinematic robot model with configurable constraints."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .environment import wrap_angle, Environment


@dataclass(slots=True)
class RobotConfig:
    radius: float = 0.3
    max_linear_velocity: float = 1.5   # m/s
    max_angular_velocity: float = 2.0  # rad/s


@dataclass(slots=True)
class RobotState:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def heading_vector(self) -> np.ndarray:
        return np.array([np.cos(self.theta), np.sin(self.theta)], dtype=np.float64)


class Robot:
    """
    Holonomic robot capable of independent x/y translation and theta rotation.
    Integrates motion using discrete-time kinematics and enforces workspace limits.
    """

    def __init__(self, state: RobotState | None = None, config: RobotConfig | None = None):
        self.state = state if state is not None else RobotState()
        self.config = config if config is not None else RobotConfig()

    def set_control(self, vx: float, vy: float, omega: float) -> None:
        """Set commanded velocities subject to saturation limits."""
        linear_speed = np.hypot(vx, vy)
        if linear_speed > self.config.max_linear_velocity:
            scale = self.config.max_linear_velocity / linear_speed
            vx *= scale
            vy *= scale

        self.state.vx = float(vx)
        self.state.vy = float(vy)
        self.state.omega = float(np.clip(
            omega, -self.config.max_angular_velocity, self.config.max_angular_velocity
        ))

    def integrate(self, dt: float, env: Environment) -> None:
        """
        Advance state by dt.
        Enforces workspace bounding box against robot radius.
        """
        # Kinematic integration
        new_x = self.state.x + self.state.vx * dt
        new_y = self.state.y + self.state.vy * dt
        new_theta = wrap_angle(self.state.theta + self.state.omega * dt)

        # Boundary collision clamping with robot radius
        r = self.config.radius
        clamped_x = np.clip(new_x, env.x_min + r, env.x_max - r)
        clamped_y = np.clip(new_y, env.y_min + r, env.y_max - r)

        # Zero out velocity component if hitting boundary
        if clamped_x != new_x:
            self.state.vx = 0.0
        if clamped_y != new_y:
            self.state.vy = 0.0

        self.state.x = float(clamped_x)
        self.state.y = float(clamped_y)
        self.state.theta = float(new_theta)