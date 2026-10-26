"""Continuous obstacle abstractions and vectorized numerical discretization for ray casting."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


def wrap_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi)."""
    return float((theta + np.pi) % (2.0 * np.pi) - np.pi)


class ContinuousObstacle(ABC):
    """
    Abstract base class for continuous 2D obstacles.
    Maintains a continuous mathematical representation and provides a method
    to produce a discrete piecewise-linear approximation for numerical ray casting.
    """

    @abstractmethod
    def discretize(self) -> np.ndarray:
        """
        Discretize the continuous curve into line segments.
        Returns:
            np.ndarray of shape (N, 2, 2) where each entry is [[x1, y1], [x2, y2]].
        """
        pass


class LineSegmentObstacle(ContinuousObstacle):
    """A finite straight line segment between point A and point B."""

    def __init__(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)

    def discretize(self) -> np.ndarray:
        return np.array([[self.p1, self.p2]], dtype=np.float64)


class FunctionObstacle(ContinuousObstacle):
    """
    Continuous function-based curve y = f(x) over [x_min, x_max].
    Approximated by dense piecewise line segments.
    """

    def __init__(
        self,
        func: Callable[[np.ndarray], np.ndarray],
        x_min: float,
        x_max: float,
        num_samples: int = 100,
    ):
        self.func = func
        self.x_min = x_min
        self.x_max = x_max
        self.num_samples = max(2, num_samples)

    def discretize(self) -> np.ndarray:
        xs = np.linspace(self.x_min, self.x_max, self.num_samples)
        ys = self.func(xs)
        points = np.column_stack((xs, ys))
        # Form line segments from adjacent points: [[p0, p1], [p1, p2], ...]
        starts = points[:-1]
        ends = points[1:]
        return np.stack((starts, ends), axis=1)


class ParametricObstacle(ContinuousObstacle):
    """
    Continuous parametric curve gamma(u) = [x(u), y(u)] for u in [u_min, u_max].
    Useful for ellipses, circles, Bezier splines, or arbitrary continuous tracks.
    """

    def __init__(
        self,
        param_func: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        u_min: float,
        u_max: float,
        num_samples: int = 100,
    ):
        self.param_func = param_func
        self.u_min = u_min
        self.u_max = u_max
        self.num_samples = max(2, num_samples)

    def discretize(self) -> np.ndarray:
        us = np.linspace(self.u_min, self.u_max, self.num_samples)
        xs, ys = self.param_func(us)
        points = np.column_stack((xs, ys))
        starts = points[:-1]
        ends = points[1:]
        return np.stack((starts, ends), axis=1)


class Environment:
    """
    Environment representation holding workspace bounds and continuous obstacles.
    Caches discretized segments into a single contiguous array for vector ray casting.
    """

    def __init__(
        self,
        x_bounds: Tuple[float, float],
        y_bounds: Tuple[float, float],
        include_boundary_as_obstacles: bool = True,
    ):
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        self.obstacles: List[ContinuousObstacle] = []
        self._cached_segments: np.ndarray | None = None
        self.include_boundary = include_boundary_as_obstacles

        if self.include_boundary:
            # 4 boundary segments: Bottom, Right, Top, Left
            b_p1 = (self.x_min, self.y_min)
            b_p2 = (self.x_max, self.y_min)
            b_p3 = (self.x_max, self.y_max)
            b_p4 = (self.x_min, self.y_max)
            self.add_obstacle(LineSegmentObstacle(b_p1, b_p2))
            self.add_obstacle(LineSegmentObstacle(b_p2, b_p3))
            self.add_obstacle(LineSegmentObstacle(b_p3, b_p4))
            self.add_obstacle(LineSegmentObstacle(b_p4, b_p1))

    def add_obstacle(self, obstacle: ContinuousObstacle) -> None:
        """Add continuous obstacle and invalidate geometry cache."""
        self.obstacles.append(obstacle)
        self._cached_segments = None

    def is_inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        """Check if a point is within the workspace boundaries with a safety margin."""
        return (
            (self.x_min + margin) <= x <= (self.x_max - margin)
            and (self.y_min + margin) <= y <= (self.y_max - margin)
        )

    def get_discretized_segments(self) -> np.ndarray:
        """
        Returns cached array of all segments in the environment.
        Shape: (M, 2, 2) where M is total segments.
        """
        if self._cached_segments is None:
            if not self.obstacles:
                self._cached_segments = np.empty((0, 2, 2), dtype=np.float64)
            else:
                segment_lists = [obs.discretize() for obs in self.obstacles]
                self._cached_segments = np.concatenate(segment_lists, axis=0)
        return self._cached_segments