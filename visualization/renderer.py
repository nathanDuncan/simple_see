"""Real-time Pygame renderer with dual sidebar inspection and point ID subscripts."""

from __future__ import annotations
from typing import List, Tuple, Any, Optional
import numpy as np
import pygame

from simulation.environment import Environment
from simulation.robot import Robot
from simulation.sensor import LidarSensor


class WorldToScreenMapper:
    def __init__(
        self,
        world_x_bounds: Tuple[float, float],
        world_y_bounds: Tuple[float, float],
        viewport_rect: Tuple[int, int, int, int],
        margin_px: int = 30,
    ):
        self.x_min, self.x_max = world_x_bounds
        self.y_min, self.y_max = world_y_bounds
        self.vx, self.vy, self.vw, self.vh = viewport_rect
        self.margin_px = margin_px

        world_w = self.x_max - self.x_min
        world_h = self.y_max - self.y_min

        usable_w = self.vw - 2 * margin_px
        usable_h = self.vh - 2 * margin_px

        self.scale = min(usable_w / world_w, usable_h / world_h)
        self.offset_x = self.vx + margin_px + (usable_w - world_w * self.scale) / 2.0
        self.offset_y = self.vy + margin_px + (usable_h - world_h * self.scale) / 2.0

    def to_screen(self, x: float, y: float) -> Tuple[int, int]:
        px = int(self.offset_x + (x - self.x_min) * self.scale)
        py = int(self.vy + self.vh - (self.offset_y - self.vy + (y - self.y_min) * self.scale))
        return px, py

    def scale_dist(self, dist_meters: float) -> int:
        return max(1, int(dist_meters * self.scale))


class SimulationRenderer:
    def __init__(
        self,
        env: Environment,
        screen_size: Tuple[int, int] = (1500, 850),
        left_sidebar_w: int = 280,
        right_sidebar_w: int = 340,
        title: str = "2D Robot Exploration Simulator - SEE",
    ):
        pygame.init()
        pygame.font.init()
        self.screen_size = screen_size
        self.left_w = left_sidebar_w
        self.right_w = right_sidebar_w
        self.screen = pygame.display.set_mode(screen_size, pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

        self.font_main = pygame.font.SysFont("Menlo", 12)
        self.font_subscript = pygame.font.SysFont("Menlo", 10, bold=True)
        self.font_bold = pygame.font.SysFont("Menlo", 12, bold=True)
        self.font_header = pygame.font.SysFont("Menlo", 14, bold=True)
        self.font_section = pygame.font.SysFont("Menlo", 12, bold=True)
        self.clock = pygame.time.Clock()

        # Center workspace viewport between left and right sidebars
        ws_width = screen_size[0] - self.left_w - self.right_w
        workspace_viewport = (self.left_w, 0, ws_width, screen_size[1])
        self.mapper = WorldToScreenMapper((env.x_min, env.x_max), (env.y_min, env.y_max), workspace_viewport)

        # Palette
        self.COLOR_BG = (15, 18, 24)
        self.COLOR_SIDEBAR_BG = (20, 24, 32)
        self.COLOR_SIDEBAR_BORDER = (45, 52, 65)
        self.COLOR_WORKSPACE = (26, 31, 40)
        self.COLOR_OBSTACLE = (115, 122, 135)
        self.COLOR_ROBOT = (65, 160, 250)
        self.COLOR_HEADING = (255, 255, 255)
        self.COLOR_RAY_PASS = (60, 100, 120, 45)
        self.COLOR_RAY_HIT = (255, 120, 60, 110)
        self.COLOR_HIT_POINT = (255, 180, 50)
        self.COLOR_TARGET = (250, 210, 60)
        self.COLOR_VISITED = (120, 110, 80)
        self.COLOR_TRAJECTORY = (70, 120, 180)
        self.COLOR_TEXT = (210, 215, 225)
        self.COLOR_LABEL = (135, 145, 160)

        # Point classifications
        self.COLOR_P_CORE = (40, 220, 110)
        self.COLOR_P_FRONTIER = (50, 150, 255)
        self.COLOR_P_OUTLIER = (245, 60, 60)

        self.show_rays: bool = True
        self.show_accumulated: bool = True
        self.show_trajectory: bool = True

    def render(
        self,
        env: Environment,
        robot: Robot,
        sensor: LidarSensor,
        current_target: Any,
        visited_waypoints: List[np.ndarray],
        trajectory: List[Tuple[float, float]],
        sim_time: float,
        is_paused: bool,
        planner: Any = None,
        detailed_animation: bool = False,
    ) -> None:
        self.screen.fill(self.COLOR_BG)

        # 1. Left Sidebar: Simulation State & Controls
        self._render_left_sidebar(robot, planner, current_target, sim_time, is_paused, detailed_animation)

        # 2. Workspace Viewport
        p_bl = self.mapper.to_screen(env.x_min, env.y_min)
        p_tr = self.mapper.to_screen(env.x_max, env.y_max)
        ws_rect = pygame.Rect(p_bl[0], p_tr[1], p_tr[0] - p_bl[0], p_bl[1] - p_tr[1])
        pygame.draw.rect(self.screen, self.COLOR_WORKSPACE, ws_rect)
        pygame.draw.rect(self.screen, (55, 65, 80), ws_rect, width=1)

        # Obstacles (Background)
        for seg in env.get_discretized_segments():
            s_start = self.mapper.to_screen(seg[0, 0], seg[0, 1])
            s_end = self.mapper.to_screen(seg[1, 0], seg[1, 1])
            pygame.draw.line(self.screen, self.COLOR_OBSTACLE, s_start, s_end, 2)

        # Waypoints
        for wp in visited_waypoints:
            wx, wy = self.mapper.to_screen(wp[0], wp[1])
            pygame.draw.circle(self.screen, self.COLOR_VISITED, (wx, wy), 4, width=1)

        if current_target is not None:
            t_pos = current_target.position if hasattr(current_target, "position") else current_target
            tx, ty = self.mapper.to_screen(float(t_pos[0]), float(t_pos[1]))
            pygame.draw.circle(self.screen, self.COLOR_TARGET, (tx, ty), 6, width=2)
            pygame.draw.line(self.screen, self.COLOR_TARGET, (tx - 8, ty), (tx + 8, ty), 1)
            pygame.draw.line(self.screen, self.COLOR_TARGET, (tx, ty - 8), (tx, ty + 8), 1)

        # Trajectory
        if self.show_trajectory and len(trajectory) > 1:
            screen_pts = [self.mapper.to_screen(pt[0], pt[1]) for pt in trajectory[-600:]]
            pygame.draw.lines(self.screen, self.COLOR_TRAJECTORY, False, screen_pts, 1)

        # Sensor Rays
        if self.show_rays and sensor.is_observing:
            ray_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)
            for hit in sensor.last_scan:
                s_orig = self.mapper.to_screen(hit.origin[0], hit.origin[1])
                if hit.detected and hit.hit_point is not None:
                    s_hit = self.mapper.to_screen(hit.hit_point[0], hit.hit_point[1])
                    pygame.draw.line(ray_surface, self.COLOR_RAY_HIT, s_orig, s_hit, 1)
                    pygame.draw.circle(self.screen, self.COLOR_HIT_POINT, s_hit, 3)
                else:
                    endpoint = hit.origin + hit.range * hit.direction
                    s_end = self.mapper.to_screen(endpoint[0], endpoint[1])
                    pygame.draw.line(ray_surface, self.COLOR_RAY_PASS, s_orig, s_end, 1)
            self.screen.blit(ray_surface, (0, 0))

        # Observed Points with Numerical Subscripts
        if planner is not None and self.show_accumulated:
            for i, pt in enumerate(planner.P_coords):
                sx, sy = self.mapper.to_screen(pt[0], pt[1])
                pt_color = (
                    self.COLOR_P_CORE if i in planner.P_c_idx
                    else self.COLOR_P_FRONTIER if i in planner.P_f_idx
                    else self.COLOR_P_OUTLIER
                )
                pygame.draw.circle(self.screen, pt_color, (sx, sy), 3)

                # Subscript label (p_i)
                txt_id = self.font_subscript.render(str(i), True, (240, 240, 245))
                self.screen.blit(txt_id, (sx + 4, sy - 11))

            # Candidate views V[f]
            for cand in getattr(planner, "proposed_views", []):
                cx, cy = self.mapper.to_screen(cand.position[0], cand.position[1])
                diamond = [(cx, cy - 5), (cx + 5, cy), (cx, cy + 5), (cx - 5, cy)]
                pygame.draw.polygon(self.screen, (255, 230, 80), diamond)
                pygame.draw.polygon(self.screen, (255, 255, 255), diamond, width=1)
                hx = cx + int(14 * np.cos(cand.heading))
                hy = cy - int(14 * np.sin(cand.heading))
                pygame.draw.line(self.screen, (255, 230, 80), (cx, cy), (hx, hy), 2)

        # Robot Body & Heading
        r_pos = robot.state.position
        s_rx, s_ry = self.mapper.to_screen(r_pos[0], r_pos[1])
        r_rad = self.mapper.scale_dist(robot.config.radius)
        pygame.draw.circle(self.screen, self.COLOR_ROBOT, (s_rx, s_ry), r_rad)
        pygame.draw.circle(self.screen, (255, 255, 255), (s_rx, s_ry), r_rad, width=2)
        h_end = r_pos + robot.config.radius * 1.35 * robot.state.heading_vector
        s_hx, s_hy = self.mapper.to_screen(h_end[0], h_end[1])
        pygame.draw.line(self.screen, self.COLOR_HEADING, (s_rx, s_ry), (s_hx, s_hy), 3)

        # Active Point Highlight in Workspace
        if detailed_animation and planner is not None and planner.current_debug_state is not None:
            self._render_workspace_active_halo(planner.current_debug_state, planner)

        # 3. Right Sidebar: Detailed Step-by-Step Classification Inspector
        self._render_right_sidebar(planner)

        pygame.display.flip()

    def _render_left_sidebar(
        self,
        robot: Robot,
        planner: Any,
        current_target: Any,
        sim_time: float,
        is_paused: bool,
        detailed_animation: bool,
    ) -> None:
        pygame.draw.rect(self.screen, self.COLOR_SIDEBAR_BG, pygame.Rect(0, 0, self.left_w, self.screen_size[1]))
        pygame.draw.line(self.screen, self.COLOR_SIDEBAR_BORDER, (self.left_w, 0), (self.left_w, self.screen_size[1]), 2)

        y = 20
        self.screen.blit(self.font_header.render("SEE EXPLORER", True, (255, 255, 255)), (18, y))
        y += 32

        mode_str = "STEP-BY-STEP" if detailed_animation else ("PAUSED" if is_paused else "RUNNING")
        mode_col = (255, 220, 80) if detailed_animation else ((255, 200, 80) if is_paused else (100, 230, 120))

        if current_target is not None:
            pos = current_target.position if hasattr(current_target, "position") else current_target
            target_pos_str = f"({pos[0]:.2f}, {pos[1]:.2f})"
            if hasattr(current_target, "heading") and current_target.heading is not None:
                target_head_str = f"{current_target.heading:.2f} rad ({int(np.degrees(current_target.heading))}°)"
            else:
                target_head_str = "None"
        else:
            target_pos_str = "None"
            target_head_str = "None"

        items = [
            ("Execution Mode", mode_str, mode_col),
            ("Sim Time", f"{sim_time:.2f} s", self.COLOR_TEXT),
            ("Robot Pose (x, y)", f"({robot.state.x:.2f}, {robot.state.y:.2f}) m", self.COLOR_TEXT),
            ("Robot Heading", f"{robot.state.theta:.2f} rad ({int(np.degrees(robot.state.theta))}°)", self.COLOR_TEXT),
            ("Target Position", target_pos_str, self.COLOR_TARGET),
            ("Target Heading", target_head_str, (255, 200, 60)),
            ("Core Points (P_c)", f"{len(planner.P_c) if planner else 0}", self.COLOR_P_CORE),
            ("Frontiers (P_f)", f"{len(planner.P_f) if planner else 0}", self.COLOR_P_FRONTIER),
            ("Outliers (P_o)", f"{len(planner.P_o) if planner else 0}", self.COLOR_P_OUTLIER),
            ("Total Points (P)", f"{len(planner.P) if planner else 0}", self.COLOR_TEXT),
        ]

        for label, val, col in items:
            self.screen.blit(self.font_main.render(label, True, self.COLOR_LABEL), (18, y))
            self.screen.blit(self.font_bold.render(val, True, col), (18, y + 15))
            y += 37

        y = self.screen_size[1] - 170
        pygame.draw.line(self.screen, self.COLOR_SIDEBAR_BORDER, (15, y - 10), (self.left_w - 15, y - 10), 1)
        self.screen.blit(self.font_bold.render("CONTROLS", True, (255, 255, 255)), (18, y))
        y += 24

        for c in ["SPACE: Step / Pause", "D: Toggle Step Mode", "R: Reset", "L: Rays | O: Obs", "T: Trajectory"]:
            self.screen.blit(self.font_main.render(c, True, (160, 170, 185)), (18, y))
            y += 18

    def _render_right_sidebar(self, planner: Any) -> None:
        rx = self.screen_size[0] - self.right_w
        pygame.draw.rect(self.screen, self.COLOR_SIDEBAR_BG, pygame.Rect(rx, 0, self.right_w, self.screen_size[1]))
        pygame.draw.line(self.screen, self.COLOR_SIDEBAR_BORDER, (rx, 0), (rx, self.screen_size[1]), 2)

        y = 20
        self.screen.blit(self.font_header.render("CLASSIFICATION INSPECTOR", True, (255, 255, 255)), (rx + 16, y))
        y += 28

        debug = getattr(planner, "current_debug_state", None)
        if debug is None:
            txt_idle = self.font_main.render("Waiting for active scan / step...", True, self.COLOR_LABEL)
            self.screen.blit(txt_idle, (rx + 16, y + 10))
            return

        # 1. Active Point Card
        p_col = (
            self.COLOR_P_CORE if debug.outcome == "Core"
            else self.COLOR_P_FRONTIER if debug.outcome == "Frontier"
            else self.COLOR_P_OUTLIER
        )
        self.screen.blit(self.font_section.render(f"ACTIVE POINT: p_{debug.active_idx}", True, (255, 230, 80)), (rx + 16, y))
        y += 20
        coord_str = f"Coords: ({debug.active_point[0]:.2f}, {debug.active_point[1]:.2f})"
        self.screen.blit(self.font_main.render(coord_str, True, self.COLOR_TEXT), (rx + 16, y))
        y += 18
        self.screen.blit(self.font_bold.render(f"Verdict: {debug.outcome.upper()}", True, p_col), (rx + 16, y))
        y += 20
        self.screen.blit(self.font_main.render(debug.status_msg, True, (200, 205, 215)), (rx + 16, y))
        y += 30

        pygame.draw.line(self.screen, self.COLOR_SIDEBAR_BORDER, (rx + 16, y), (self.screen_size[0] - 16, y), 1)
        y += 12

        # 2. Neighbors Card (N_q)
        self.screen.blit(
            self.font_section.render(f"NEIGHBOURS N_{debug.active_idx} (r={debug.radius:.2f}m)", True, (255, 255, 255)), 
            (rx + 16, y)
        )
        y += 20
        
        # Display the position-dependent threshold
        k_req = getattr(debug, "effective_k_min", 3)
        condition_tag = "2x (x > 0)" if debug.active_point[0] > 0.0 else "1x (x <= 0)"
        thresh_str = f"Threshold k_min: {k_req} [{condition_tag}]"
        self.screen.blit(self.font_bold.render(thresh_str, True, (255, 220, 80)), (rx + 16, y))
        y += 18

        n_summary = f"Total |N_q|: {debug.total_neighbors} | Core: {debug.num_core_neighbors}"
        self.screen.blit(self.font_main.render(n_summary, True, (170, 180, 195)), (rx + 16, y))
        y += 20

        # Neighbor list
        for n_id in debug.neighbor_indices[:8]:
            n_class = debug.neighbor_classes.get(n_id, "Unknown")
            n_col = (
                self.COLOR_P_CORE if n_class == "Core"
                else self.COLOR_P_FRONTIER if n_class == "Frontier"
                else self.COLOR_P_OUTLIER
            )
            txt_n = self.font_main.render(f"• p_{n_id}: {n_class}", True, n_col)
            self.screen.blit(txt_n, (rx + 24, y))
            y += 17

        if len(debug.neighbor_indices) > 8:
            txt_more = self.font_main.render(f"... +{len(debug.neighbor_indices) - 8} more", True, self.COLOR_LABEL)
            self.screen.blit(txt_more, (rx + 24, y))
            y += 18

        y += 12
        pygame.draw.line(self.screen, self.COLOR_SIDEBAR_BORDER, (rx + 16, y), (self.screen_size[0] - 16, y), 1)
        y += 12

        # 3. Processing Queue (Q)
        self.screen.blit(self.font_section.render(f"QUEUE Q (Remaining: {len(debug.queue_indices)})", True, (255, 255, 255)), (rx + 16, y))
        y += 20

        if not debug.queue_indices:
            self.screen.blit(self.font_main.render("(Queue empty)", True, self.COLOR_LABEL), (rx + 16, y))
        else:
            q_str = " -> ".join([f"p_{idx}" for idx in debug.queue_indices[:12]])
            if len(debug.queue_indices) > 12:
                q_str += " ..."
            # Render queue chips
            words = q_str.split(" -> ")
            line_str = ""
            for w in words:
                if len(line_str + w) > 28:
                    self.screen.blit(self.font_main.render(line_str, True, (255, 210, 110)), (rx + 20, y))
                    y += 18
                    line_str = "-> " + w + " "
                else:
                    line_str += ("-> " if line_str else "") + w + " "
            if line_str:
                self.screen.blit(self.font_main.render(line_str, True, (255, 210, 110)), (rx + 20, y))

    def _render_workspace_active_halo(self, debug: Any, planner: Any) -> None:
        qx, qy = self.mapper.to_screen(debug.active_point[0], debug.active_point[1])
        r_px = self.mapper.scale_dist(debug.radius)

        # Translucent radius halo
        halo = pygame.Surface((r_px * 2 + 4, r_px * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(halo, (255, 240, 100, 35), (r_px + 2, r_px + 2), r_px)
        pygame.draw.circle(halo, (255, 220, 80, 180), (r_px + 2, r_px + 2), r_px, width=2)
        self.screen.blit(halo, (qx - r_px - 2, qy - r_px - 2))

        # Connector lines to valid neighbors
        if planner is not None and hasattr(planner, "P_coords"):
            for n_id in debug.neighbor_indices:
                if 0 <= n_id < len(planner.P_coords):
                    n_pt = planner.P_coords[n_id]
                    nx, ny = self.mapper.to_screen(n_pt[0], n_pt[1])
                    pygame.draw.line(self.screen, (255, 230, 120, 160), (qx, qy), (nx, ny), 1)

        # Target circle on active point
        pygame.draw.circle(self.screen, (255, 255, 0), (qx, qy), 7, width=2)