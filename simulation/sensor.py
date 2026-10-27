"""2D LiDAR simulation with instantaneous full-scan capture."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from .environment import wrap_angle, Environment
from .robot import RobotState


@dataclass(slots=True)
class RayHit:
    """Result of a single LiDAR beam intersection test."""
    ray_index: int
    angle_rad: float               # Ray angle in world frame
    origin: np.ndarray             # [x, y]
    direction: np.ndarray          # Unit vector [dx, dy]
    range: float                   # Measured range in meters
    hit_point: Optional[np.ndarray] # [x, y] in world frame if detected, else None
    detected: bool                 # True if valid obstacle intersection within max_range


@dataclass(slots=True)
class LidarConfig:
    num_rays: int = 7
    fov_rad: float = 0.5 * np.pi   # Full 360-degree scan
    max_range: float = 4.5         # Meters
    offset_angle: float = 0.0      # Angular mounting offset relative to robot
    noise_std: float = 0.000       # Range measurement noise (0.0 for deterministic)


class LidarSensor:
    """
    Simulates an instantaneous 2D LiDAR.
    Fires all rays simultaneously and aggregates hits into observation history.
    """

    def __init__(self, config: LidarConfig | None = None, display_duration: float = 0.35):
        self.config = config if config is not None else LidarConfig()
        self.display_duration = display_duration  # Seconds rays remain visible after a scan
        self.display_timer: float = 0.0
        self._last_scan: List[RayHit] = []
        self._accumulated_hits: List[np.ndarray] = []

    @property
    def is_observing(self) -> bool:
        """True only during the active observation display window."""
        return self.display_timer > 0.0

    @property
    def last_scan(self) -> List[RayHit]:
        return self._last_scan

    @property
    def accumulated_hits(self) -> List[np.ndarray]:
        return self._accumulated_hits

    def clear_accumulated_hits(self) -> None:
        self._accumulated_hits.clear()
        self._last_scan.clear()
        self.display_timer = 0.0

    def update(self, dt: float) -> None:
        """Tick down the display duration for the visual flash."""
        if self.display_timer > 0.0:
            self.display_timer = max(0.0, self.display_timer - dt)

    def scan(self, robot_state: RobotState, env: Environment) -> List[RayHit]:
        """
        Cast all rays simultaneously from current pose, stamp hit points,
        and trigger the visual observation window.
        """
        all_hits = self._cast_all_rays(robot_state, env)
        self._last_scan = all_hits
        self.display_timer = self.display_duration

        for hit in all_hits:
            if hit.detected and hit.hit_point is not None:
                self._accumulated_hits.append(hit.hit_point.copy())

        return self._last_scan

    def compute_ray_angles(self, robot_theta: float) -> np.ndarray:
        base_angle = wrap_angle(robot_theta + self.config.offset_angle)
        if np.isclose(self.config.fov_rad, 2.0 * np.pi):
            angles_rel = np.linspace(-np.pi, np.pi, self.config.num_rays, endpoint=False)
        else:
            half_fov = self.config.fov_rad / 2.0
            angles_rel = np.linspace(-half_fov, half_fov, self.config.num_rays, endpoint=True)
        return (base_angle + angles_rel + np.pi) % (2.0 * np.pi) - np.pi

    def _cast_all_rays(self, robot_state: RobotState, env: Environment) -> List[RayHit]:
        angles = self.compute_ray_angles(robot_state.theta)
        origin = robot_state.position
        directions = np.column_stack((np.cos(angles), np.sin(angles)))

        segments = env.get_discretized_segments()
        M = segments.shape[0]
        ray_hits: List[RayHit] = []

        if M == 0:
            for i, (angle, direction) in enumerate(zip(angles, directions)):
                ray_hits.append(
                    RayHit(
                        ray_index=i,
                        angle_rad=float(angle),
                        origin=origin.copy(),
                        direction=direction.copy(),
                        range=self.config.max_range,
                        hit_point=None,
                        detected=False,
                    )
                )
            return ray_hits

        # Broadcast vector intersection across all M segments
        A = segments[:, 0, :]
        B = segments[:, 1, :]
        V = B - A

        D_exp = directions[:, np.newaxis, :]       # (K, 1, 2)
        V_exp = V[np.newaxis, :, :]                # (1, M, 2)
        A_exp = A[np.newaxis, :, :]                # (1, M, 2)
        P_exp = origin[np.newaxis, np.newaxis, :]  # (1, 1, 2)
        W_exp = A_exp - P_exp                      # (1, M, 2)

        denom = D_exp[:, :, 0] * V_exp[:, :, 1] - D_exp[:, :, 1] * V_exp[:, :, 0]
        num_t = W_exp[:, :, 0] * V_exp[:, :, 1] - W_exp[:, :, 1] * V_exp[:, :, 0]
        num_u = W_exp[:, :, 0] * D_exp[:, :, 1] - W_exp[:, :, 1] * D_exp[:, :, 0]

        non_zero = np.abs(denom) > 1e-9
        inv_denom = np.zeros_like(denom)
        inv_denom[non_zero] = 1.0 / denom[non_zero]

        t = num_t * inv_denom
        u = num_u * inv_denom

        valid = non_zero & (u >= 0.0) & (u <= 1.0) & (t >= 0.0) & (t <= self.config.max_range)

        inf_dist = self.config.max_range + 10.0
        t_masked = np.where(valid, t, inf_dist)
        min_t_indices = np.argmin(t_masked, axis=1)

        for i in range(len(angles)):
            best_seg_idx = min_t_indices[i]
            is_hit = valid[i, best_seg_idx]
            dist = float(t[i, best_seg_idx]) if is_hit else self.config.max_range
            angle = float(angles[i])
            dir_vec = directions[i]

            if is_hit:
                if self.config.noise_std > 0.0:
                    dist = float(np.clip(
                        dist + np.random.normal(0.0, self.config.noise_std),
                        0.0,
                        self.config.max_range,
                    ))
                hit_pt = origin + dist * dir_vec
            else:
                hit_pt = None

            ray_hits.append(
                RayHit(
                    ray_index=i,
                    angle_rad=angle,
                    origin=origin.copy(),
                    direction=dir_vec.copy(),
                    range=dist,
                    hit_point=hit_pt,
                    detected=is_hit,
                )
            )

        return ray_hits