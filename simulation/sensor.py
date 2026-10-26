"""2D LiDAR simulation with vectorized ray casting and progressive scanning."""

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
    num_rays: int = 120
    fov_rad: float = 2.0 * np.pi   # Full 360-degree scan by default
    max_range: float = 5.0         # Meters
    offset_angle: float = 0.0      # Angular mounting offset relative to robot
    noise_std: float = 0.0         # Optional Gaussian range measurement noise
    sweep_rate_hz: float = 2.0     # Sweeps per second for progressive visualization


class LidarSensor:
    """
    Simulates a 2D LiDAR.
    Performs vectorized ray-segment intersection against cached environment segments.
    Maintains progressive sweep state for smooth visual scanning.
    """

    def __init__(self, config: LidarConfig | None = None):
        self.config = config if config is not None else LidarConfig()
        self._sweep_progress: float = 0.0  # Fraction [0.0, 1.0]
        self._last_scan: List[RayHit] = []
        self._accumulated_hits: List[np.ndarray] = []

    @property
    def sweep_progress(self) -> float:
        return self._sweep_progress

    @property
    def last_scan(self) -> List[RayHit]:
        return self._last_scan

    @property
    def accumulated_hits(self) -> List[np.ndarray]:
        return self._accumulated_hits

    def clear_accumulated_hits(self) -> None:
        self._accumulated_hits.clear()

    def update_sweep(self, robot_state: RobotState, dt: float, env: Environment) -> None:
        """
        Advance LiDAR sweep progress and perform ray-casting across the active beam sector.
        """
        prev_progress = self._sweep_progress
        self._sweep_progress += self.config.sweep_rate_hz * dt

        if self._sweep_progress >= 1.0:
            # Complete the sweep, generate remaining rays, reset sweep progress
            self._sweep_progress %= 1.0
            full_scan = self.cast_all_rays(robot_state, env)
            self._last_scan = full_scan

            # Accumulate newly detected obstacle points
            for hit in full_scan:
                if hit.detected and hit.hit_point is not None:
                    self._accumulated_hits.append(hit.hit_point.copy())
        else:
            # In-progress partial update: update scan rays up to current sweep index
            max_idx = int(self._sweep_progress * self.config.num_rays)
            partial_scan = self.cast_rays_slice(robot_state, env, 0, max_idx)
            self._last_scan = partial_scan

    def compute_ray_angles(self, robot_theta: float) -> np.ndarray:
        """Calculate world-frame angles for all sensor rays."""
        base_angle = wrap_angle(robot_theta + self.config.offset_angle)
        half_fov = self.config.fov_rad / 2.0
        if np.isclose(self.config.fov_rad, 2.0 * np.pi):
            # Uniform circle without duplicating the start/end ray
            angles_rel = np.linspace(-np.pi, np.pi, self.config.num_rays, endpoint=False)
        else:
            angles_rel = np.linspace(-half_fov, half_fov, self.config.num_rays, endpoint=True)
        return (base_angle + angles_rel + np.pi) % (2.0 * np.pi) - np.pi

    def cast_all_rays(self, robot_state: RobotState, env: Environment) -> List[RayHit]:
        return self.cast_rays_slice(robot_state, env, 0, self.config.num_rays)

    def cast_rays_slice(
        self, robot_state: RobotState, env: Environment, start_idx: int, end_idx: int
    ) -> List[RayHit]:
        """
        Vectorized ray-casting against all environment segments.
        Calculates line-segment intersection using 2D cross-products.
        """
        all_angles = self.compute_ray_angles(robot_state.theta)
        end_idx = min(end_idx, self.config.num_rays)
        if start_idx >= end_idx:
            return []

        angles = all_angles[start_idx:end_idx]
        num_slice_rays = len(angles)
        origin = robot_state.position  # shape (2,)

        # Unit direction vectors: shape (K, 2)
        directions = np.column_stack((np.cos(angles), np.sin(angles)))

        segments = env.get_discretized_segments()  # shape (M, 2, 2)
        M = segments.shape[0]

        ray_hits: List[RayHit] = []

        if M == 0:
            for i, (angle, direction) in enumerate(zip(angles, directions)):
                ray_hits.append(
                    RayHit(
                        ray_index=start_idx + i,
                        angle_rad=float(angle),
                        origin=origin.copy(),
                        direction=direction.copy(),
                        range=self.config.max_range,
                        hit_point=None,
                        detected=False,
                    )
                )
            return ray_hits

        # Segment vectors: A, B shape (M, 2)
        A = segments[:, 0, :]
        B = segments[:, 1, :]
        V = B - A  # Segment direction vectors (M, 2)

        # Broadcasting dimensions:
        # Rays: (K, 1, 2)
        # Segments: (1, M, 2)
        D_exp = directions[:, np.newaxis, :]  # (K, 1, 2)
        V_exp = V[np.newaxis, :, :]           # (1, M, 2)
        A_exp = A[np.newaxis, :, :]           # (1, M, 2)
        P_exp = origin[np.newaxis, np.newaxis, :]  # (1, 1, 2)
        W_exp = A_exp - P_exp                 # (1, M, 2)

        # 2D cross products:
        # denom = D x V = D_x * V_y - D_y * V_x
        denom = D_exp[:, :, 0] * V_exp[:, :, 1] - D_exp[:, :, 1] * V_exp[:, :, 0]

        # Numerators for t and u:
        # t * denom = W x V = W_x * V_y - W_y * V_x
        # u * denom = W x D = W_x * D_y - W_y * D_x
        num_t = W_exp[:, :, 0] * V_exp[:, :, 1] - W_exp[:, :, 1] * V_exp[:, :, 0]
        num_u = W_exp[:, :, 0] * D_exp[:, :, 1] - W_exp[:, :, 1] * D_exp[:, :, 0]

        # Avoid zero division
        non_zero = np.abs(denom) > 1e-9
        inv_denom = np.zeros_like(denom)
        inv_denom[non_zero] = 1.0 / denom[non_zero]

        t = num_t * inv_denom
        u = num_u * inv_denom

        # Valid intersection criteria:
        # 1. Non-parallel rays
        # 2. Segment parameter u in [0, 1]
        # 3. Ray parameter t in [0, max_range]
        valid = non_zero & (u >= 0.0) & (u <= 1.0) & (t >= 0.0) & (t <= self.config.max_range)

        # Nearest hit selection
        inf_dist = self.config.max_range + 10.0
        t_masked = np.where(valid, t, inf_dist)
        min_t_indices = np.argmin(t_masked, axis=1)  # shape (K,)

        for i in range(num_slice_rays):
            best_seg_idx = min_t_indices[i]
            is_hit = valid[i, best_seg_idx]
            dist = float(t[i, best_seg_idx]) if is_hit else self.config.max_range
            angle = float(angles[i])
            dir_vec = directions[i]

            if is_hit:
                # Add optional measurement noise
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
                    ray_index=start_idx + i,
                    angle_rad=angle,
                    origin=origin.copy(),
                    direction=dir_vec.copy(),
                    range=dist,
                    hit_point=hit_pt,
                    detected=is_hit,
                )
            )

        return ray_hits