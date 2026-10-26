"""Real-time Pygame renderer with coordinate transformation and HUD overlays."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import pygame

from simulation.environment import Environment
from simulation.robot import Robot
from simulation.sensor import LidarSensor


class WorldToScreenMapper:
    """Handles mapping between world coordinates (meters) and Pygame screen coordinates (pixels)."""

    def __init__(
        self,
        world_x_bounds: Tuple[float, float],
        world_y_bounds: Tuple[float, float],
        screen_size: Tuple[int, int],
        margin_px: int = 40,
    ):
        self.x_min, self.x_max = world_x_bounds
        self.y_min, self.y_max = world_y_bounds
        self.screen_width, self.screen_height = screen_size
        self.margin_px = margin_px

        world_w = self.x_max - self.x_min
        world_h = self.y_max - self.y_min

        usable_w = self.screen_width - 2 * margin_px
        usable_h = self.screen_height - 2 * margin_px

        # Preserve uniform aspect ratio
        self.scale = min(usable_w / world_w, usable_h / world_h)
        self.offset_x = margin_px + (usable_w - world_w * self.scale) / 2.0
        self.offset_y = margin_px + (usable_h - world_h * self.scale) / 2.0

    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters, Y-up) to screen coordinates (pixels, Y-down)."""
        px = int(self.offset_x + (x - self.x_min) * self.scale)
        py = int(self.screen_height - (self.offset_y + (y - self.y_min) * self.scale))
        return px, py

    def scale_dist(self, dist_meters: float) -> int:
        return max(1, int(dist_meters * self.scale))


class SimulationRenderer:
    """Detailed visualization pipeline for robot, workspace, sensor sweeps, and overlays."""

    def __init__(
        self,
        env: Environment,
        screen_size: Tuple[int, int] = (1000, 750),
        title: str = "2D Robot Exploration Simulator",
    ):
        pygame.init()
        pygame.font.init()
        self.screen_size = screen_size
        self.screen = pygame.display.set_mode(screen_size, pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

        self.font_main = pygame.font.SysFont("Menlo", 13)
        self.font_bold = pygame.font.SysFont("Menlo", 13, bold=True)
        self.clock = pygame.time.Clock()

        self.mapper = WorldToScreenMapper(
            (env.x_min, env.x_max), (env.y_min, env.y_max), screen_size
        )

        # Visual color palette
        self.COLOR_BG = (18, 22, 28)
        self.COLOR_WORKSPACE = (32, 38, 48)
        self.COLOR_OBSTACLE = (230, 90, 70)
        self.COLOR_ROBOT = (65, 160, 250)
        self.COLOR_HEADING = (255, 255, 255)
        self.COLOR_RAY_PASS = (60, 100, 120, 60)
        self.COLOR_RAY_HIT = (255, 120, 60, 140)
        self.COLOR_HIT_POINT = (255, 80, 50)
        self.COLOR_ACCUMULATED = (80, 220, 160)
        self.COLOR_TARGET = (250, 210, 60)
        self.COLOR_VISITED = (120, 110, 80)
        self.COLOR_TRAJECTORY = (70, 120, 180)
        self.COLOR_PANEL_BG = (24, 28, 36, 210)
        self.COLOR_TEXT = (220, 225, 230)

        # Toggles
        self.show_rays: bool = True
        self.show_accumulated: bool = True
        self.show_trajectory: bool = True

    def render(
        self,
        env: Environment,
        robot: Robot,
        sensor: LidarSensor,
        current_target: np.ndarray | None,
        visited_waypoints: List[np.ndarray],
        trajectory: List[Tuple[float, float]],
        sim_time: float,
        is_paused: bool,
    ) -> None:
        """Render complete frame without mutating simulation state."""
        self.screen.fill(self.COLOR_BG)

        # 1. Workspace boundary fill
        p_bl = self.mapper.to_screen(env.x_min, env.y_min)
        p_tr = self.mapper.to_screen(env.x_max, env.y_max)
        ws_rect = pygame.Rect(p_bl[0], p_tr[1], p_tr[0] - p_bl[0], p_bl[1] - p_tr[1])
        pygame.draw.rect(self.screen, self.COLOR_WORKSPACE, ws_rect)
        pygame.draw.rect(self.screen, (70, 80, 95), ws_rect, width=1)

        # 2. Accumulated observation points
        if self.show_accumulated and sensor.accumulated_hits:
            for pt in sensor.accumulated_hits:
                sx, sy = self.mapper.to_screen(pt[0], pt[1])
                pygame.draw.circle(self.screen, self.COLOR_ACCUMULATED, (sx, sy), 2)

        # 3. Obstacle continuous curves (rendered via discretized segments)
        segments = env.get_discretized_segments()
        for seg in segments:
            s_start = self.mapper.to_screen(seg[0, 0], seg[0, 1])
            s_end = self.mapper.to_screen(seg[1, 0], seg[1, 1])
            pygame.draw.line(self.screen, self.COLOR_OBSTACLE, s_start, s_end, 2)

        # 4. Waypoints
        for wp in visited_waypoints:
            wx, wy = self.mapper.to_screen(wp[0], wp[1])
            pygame.draw.circle(self.screen, self.COLOR_VISITED, (wx, wy), 4, width=1)

        if current_target is not None:
            tx, ty = self.mapper.to_screen(current_target[0], current_target[1])
            pygame.draw.circle(self.screen, self.COLOR_TARGET, (tx, ty), 6, width=2)
            pygame.draw.line(self.screen, self.COLOR_TARGET, (tx - 8, ty), (tx + 8, ty), 1)
            pygame.draw.line(self.screen, self.COLOR_TARGET, (tx, ty - 8), (tx, ty + 8), 1)

        # 5. Robot trajectory
        if self.show_trajectory and len(trajectory) > 1:
            screen_pts = [self.mapper.to_screen(pt[0], pt[1]) for pt in trajectory[-600:]]
            pygame.draw.lines(self.screen, self.COLOR_TRAJECTORY, False, screen_pts, 1)

        # 6. Active sensor sweep and rays
        r_pos = robot.state.position
        s_rx, s_ry = self.mapper.to_screen(r_pos[0], r_pos[1])

        if self.show_rays:
            ray_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            for hit in sensor.last_scan:
                if hit.detected and hit.hit_point is not None:
                    s_hit = self.mapper.to_screen(hit.hit_point[0], hit.hit_point[1])
                    pygame.draw.line(ray_surface, self.COLOR_RAY_HIT, (s_rx, s_ry), s_hit, 1)
                    pygame.draw.circle(self.screen, self.COLOR_HIT_POINT, s_hit, 3)
                else:
                    endpoint = hit.origin + hit.range * hit.direction
                    s_end = self.mapper.to_screen(endpoint[0], endpoint[1])
                    pygame.draw.line(ray_surface, self.COLOR_RAY_PASS, (s_rx, s_ry), s_end, 1)
            self.screen.blit(ray_surface, (0, 0))

        # 7. Robot body and heading
        r_radius_px = self.mapper.scale_dist(robot.config.radius)
        pygame.draw.circle(self.screen, self.COLOR_ROBOT, (s_rx, s_ry), r_radius_px)
        pygame.draw.circle(self.screen, (255, 255, 255), (s_rx, s_ry), r_radius_px, width=2)

        # Heading indicator line
        head_end = r_pos + robot.config.radius * 1.3 * robot.state.heading_vector
        s_hx, s_hy = self.mapper.to_screen(head_end[0], head_end[1])
        pygame.draw.line(self.screen, self.COLOR_HEADING, (s_rx, s_ry), (s_hx, s_hy), 3)

        # 8. HUD & Diagnostics Overlay
        self._render_hud(robot, sensor, current_target, sim_time, is_paused)

        pygame.display.flip()

    def _render_hud(
        self,
        robot: Robot,
        sensor: LidarSensor,
        current_target: np.ndarray | None,
        sim_time: float,
        is_paused: bool,
    ) -> None:
        """Render floating diagnostic telemetry and keyboard command helper."""
        panel_w, panel_h = 285, 205
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(self.COLOR_PANEL_BG)
        self.screen.blit(panel, (12, 12))
        pygame.draw.rect(self.screen, (60, 70, 85), pygame.Rect(12, 12, panel_w, panel_h), width=1)

        speed = float(np.hypot(robot.state.vx, robot.state.vy))
        target_str = f"({current_target[0]:.2f}, {current_target[1]:.2f})" if current_target is not None else "None"
        state_str = "PAUSED" if is_paused else "RUNNING"

        lines = [
            ("STATUS", state_str, (255, 200, 80) if is_paused else (100, 230, 120)),
            ("Sim Time", f"{sim_time:.2f} s", self.COLOR_TEXT),
            ("Pose (x, y)", f"({robot.state.x:.2f}, {robot.state.y:.2f}) m", self.COLOR_TEXT),
            ("Heading (θ)", f"{robot.state.theta:.2f} rad ({np.degrees(robot.state.theta):.1f}°)", self.COLOR_TEXT),
            ("Speed", f"{speed:.2f} m/s", self.COLOR_TEXT),
            ("Target WP", target_str, self.COLOR_TEXT),
            ("Sweep Prog", f"{int(sensor.sweep_progress * 100):3d}%", self.COLOR_TEXT),
            ("Points Mapped", f"{len(sensor.accumulated_hits):d}", self.COLOR_TEXT),
        ]

        y_offset = 18
        for label, val, color in lines:
            txt_lbl = self.font_main.render(f"{label}:", True, (150, 160, 175))
            txt_val = self.font_bold.render(val, True, color)
            self.screen.blit(txt_lbl, (20, y_offset))
            self.screen.blit(txt_val, (135, y_offset))
            y_offset += 22

        # Keyboard control banner in bottom left
        cmd_text = "SPACE: Pause/Resume | R: Reset | L: Rays | O: Obs | T: Trajectory"
        lbl_cmd = self.font_main.render(cmd_text, True, (140, 150, 165))
        self.screen.blit(lbl_cmd, (15, self.screen_size[1] - 25))