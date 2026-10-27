"""
Implementation of the Surface Edge Explorer (SEE) algorithm.
Based on Algorithm 1: SEE(v_c, Gamma_see, Gamma_sensor).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Dict, Tuple, Any, Generator
from collections import deque
import numpy as np

from .environment import Environment
from .robot import Robot, RobotState
from .sensor import RayHit, LidarSensor, LidarConfig
from .planner import BasePlanner


@dataclass
class SEEParameters:
    """Algorithm parameters for SEE."""
    epsilon: float = 0.08       # Separation threshold to reject duplicate points
    r_neighbor: float = 0.35    # Neighborhood radius r
    k_min: int = 5              # Minimum neighbors for core closure
    d_standoff: float = 1.6     # d: Standoff distance along e_n for proposed views
    v_step: float = 0.12        # v: Step size used in DirectNormal
    max_direct_iters: int = 15  # Loop cutoff for DirectNormal repeat-until

@dataclass
class SurfaceFrame2D:
    """E_surf[f] = (e_n, e_f, e_b) in 2D plane with e_b pointing out of screen."""
    e_n: np.ndarray             # Normal vector [nx, ny]
    e_f: np.ndarray             # Frontier tangent vector [fx, fy]
    e_b: float = 1.0            # Out-of-screen z-component (+1.0)

@dataclass
class Viewpoint:
    """Representation of an element in candidate views V or v_c."""
    position: np.ndarray        # [x, y]
    heading: float = 0.0        # Orientation in radians
    cost: float = 0.0
    utility: float = 0.0
    frontier_idx: Optional[int] = None


@dataclass
class ClassificationStepDebug:
    """State emitted during detailed step-by-step point classification."""
    active_point: np.ndarray             # Current point q being checked
    active_idx: int                      # Index of point q
    radius: float                        # Neighborhood radius r
    neighbor_indices: List[int]
    neighbor_classes: Dict[int, str]
    queue_indices: List[int]
    num_core_neighbors: int              # Number of neighbors belonging to P_c
    total_neighbors: int                 # Total |N_q|
    effective_k_min: int
    outcome: str                         # 'core', 'frontier', or 'outlier'
    status_msg: str                      # Human-readable HUD status string


class SEEPlanner(BasePlanner):
    """
    Surface Edge Explorer (SEE) planner.
    Integrates with the simulator via select_next_waypoint(...).
    """

    def __init__(self, gamma_see: Optional[SEEParameters] = None, gamma_sensor: Optional[LidarConfig] = None):
        self.gamma_see = gamma_see if gamma_see is not None else SEEParameters()
        self.gamma_sensor = gamma_sensor if gamma_sensor is not None else LidarConfig()

        # --- Lines 1-3: State Initialization ---
        # P = P_c U P_f U P_o
        # Points storage: master coordinate list indexed by int ID
        self.P_coords: List[np.ndarray] = []

        self.P_c_idx: Set[int] = set()        # P_c: Closed / completed point IDs
        self.P_f_idx: Set[int] = set()        # P_f: Frontier point IDs
        self.P_o_idx: Set[int] = set()        # P_o: Occupied / open point IDs

        self.P_prime: List[np.ndarray] = []   # P': New measurements from current step

        self.X_cap: Dict[int, np.ndarray] = {}      # X_cap: Sensor visibility / capture region
        self.V_map: Dict[int, Viewpoint] = {}       # V[f]: Map from frontier point index -> Viewpoint
        self.E_surf: Dict[int, SurfaceFrame2D] = {} # E_surf: Surface estimate / envelope

        self.f_c: Optional[np.ndarray] = None       # f_c: Current focus frontier point
        self.G: Any = None                    # G: View graph

        # Line 4: ComputeParameters(Gamma_see, Gamma_sensor)
        self.compute_parameters(self.gamma_see, self.gamma_sensor)

        self.is_complete: bool = False

        # Generator for step-by-step interactive inspection
        self.active_classifier_gen: Optional[Generator[ClassificationStepDebug, None, None]] = None
        self.current_debug_state: Optional[ClassificationStepDebug] = None

    # -------------------------------------------------------------------------
    # Properties for coordinate access
    # -------------------------------------------------------------------------

    @property
    def P(self) -> List[np.ndarray]:
        """P = P_c U P_f U P_o"""
        return self.P_coords

    @property
    def P_c(self) -> List[np.ndarray]:
        return [self.P_coords[i] for i in self.P_c_idx]

    @property
    def P_f(self) -> List[np.ndarray]:
        return [self.P_coords[i] for i in self.P_f_idx]

    @property
    def P_o(self) -> List[np.ndarray]:
        return [self.P_coords[i] for i in self.P_o_idx]

    @property
    def proposed_views(self) -> List[Viewpoint]:
        return list(self.V_map.values())

    def get_point_class(self, idx: int) -> str:
        if idx in self.P_c_idx:
            return "Core"
        if idx in self.P_f_idx:
            return "Frontier"
        if idx in self.P_o_idx:
            return "Outlier"
        return "Unclassified"

    # -------------------------------------------------------------------------
    # Geometric helper: Neighbours_r(P, radius, query_pt)
    # -------------------------------------------------------------------------

    def _find_neighbor_indices(self, query_pt: np.ndarray, radius: float) -> List[int]:
        """Returns indices of points in self.P_coords within radius of query_pt."""
        if not self.P_coords:
            return []
        coords_arr = np.array(self.P_coords)  # Shape (N, 2)
        dists = np.linalg.norm(coords_arr - query_pt, axis=1)
        return [int(idx) for idx in np.where(dists <= radius)[0]]

    # -------------------------------------------------------------------------
    # Algorithm Subfunction Placeholders
    # -------------------------------------------------------------------------

    def compute_parameters(self, gamma_see: SEEParameters, gamma_sensor: LidarConfig) -> None:
        """Line 4: ComputeParameters(Gamma_see, Gamma_sensor)"""
        # Placeholder: compute sensor footprint limits, ray spans, clustering thresholds
        pass

    def get_new_measurements(self, v_c: Viewpoint, observations: List[RayHit]) -> None:
        """
        Line 6: GetNewMeasurements(v_c, P')
        Populates P' with detected surface points.
        """
        self.P_prime.clear()
        for hit in observations:
            if hit.detected and hit.hit_point is not None:
                self.P_prime.append(hit.hit_point.copy())

    def classify_points_step_by_step(self, v_c: Viewpoint) -> Generator[ClassificationStepDebug, None, None]:
        x_c = v_c.position.copy()
        eps = self.gamma_see.epsilon
        r = self.gamma_see.r_neighbor
        k_min = self.gamma_see.k_min
        ### Line 2 ###
        R: Set[int] = set()

        ### Line 3 ###
        for p in self.P_prime:
            eps_neighbors = self._find_neighbor_indices(p, eps)
            ### Line 4 ###
            if len(eps_neighbors) == 0:
                ### Line 5 ###
                new_idx = len(self.P_coords)
                self.P_coords.append(p.copy())

                # Line 6: Q <- {p} U Neighbours_r(P, r, p)
                # Ensure p is strictly first in the queue:
                r_neighbors = self._find_neighbor_indices(p, r)

                Q: deque[int] = deque([new_idx])
                Q_set: Set[int] = {new_idx}

                for n_idx in r_neighbors:
                    if n_idx != new_idx and n_idx not in Q_set:
                        Q.append(n_idx)
                        Q_set.add(n_idx)

                self.X_cap[new_idx] = x_c.copy()

                while Q:
                    q = Q.popleft()
                    Q_set.discard(q)

                    # Line 9: if q in P_c, show that it was skipped because it's already Core!
                    if q in self.P_c_idx:
                        yield ClassificationStepDebug(
                            active_point=q_coord.copy(),
                            active_idx=q,
                            radius=r,
                            neighbor_indices=sorted(list(N_q)),
                            neighbor_classes=n_classes,
                            queue_indices=list(Q),
                            num_core_neighbors=len(core_neighbors),
                            total_neighbors=len(N_q),
                            effective_k_min=effective_k_min,
                            outcome=outcome,
                            status_msg=msg,
                        )
                        continue

                    # Dynamic k_min condition based on x coordinate
                    q_coord = self.P_coords[q]
                    effective_k_min = 2 * k_min if q_coord[0] > 0.0 else k_min

                    # Line 10: N_q <- Neighbours_r(P, r, q)
                    N_q = set(self._find_neighbor_indices(self.P_coords[q], r))
                    core_neighbors = N_q.intersection(self.P_c_idx)

                    # Line 11: if |N_q| < effective_k_min then
                    if len(N_q) < effective_k_min:
                        # Line 12: if N_q intersect P_c != empty then
                        if len(core_neighbors) > 0:
                            self.P_f_idx.add(q)
                            self.P_o_idx.discard(q)
                            outcome = "Frontier"
                            msg = f"|N_q|={len(N_q)} < {effective_k_min}, Core Neighbours={len(core_neighbors)} -> FRONTIER"
                        else:
                            self.P_o_idx.add(q)
                            outcome = "Outlier"
                            msg = f"|N_q|={len(N_q)} < {effective_k_min}, Core Neighbours=0 -> OUTLIER"
                    else:
                        self.P_c_idx.add(q)
                        self.P_f_idx.discard(q)
                        self.P_o_idx.discard(q)
                        self.V_map.pop(q, None)
                        self.E_surf.pop(q, None)
                        outcome = "Core"
                        msg = f"|N_q|={len(N_q)} >= {k_min} -> CORE"

                        # Line 28: if p != q and q not in R then Q <- Q U N_q
                        if q != new_idx and q not in R:
                            for n_idx in N_q:
                                if n_idx not in Q_set:
                                    Q_set.add(n_idx)
                                    Q.append(n_idx)
                            R.add(q)

                    n_classes = {idx: self.get_point_class(idx) for idx in N_q}

                    yield ClassificationStepDebug(
                        active_point=q_coord.copy(),
                        active_idx=q,
                        radius=r,
                        neighbor_indices=sorted(list(N_q)),
                        neighbor_classes=n_classes,
                        queue_indices=list(Q),
                        num_core_neighbors=len(core_neighbors),
                        total_neighbors=len(N_q),
                        effective_k_min=effective_k_min,
                        outcome=outcome,
                        status_msg=msg,
                    )
    
    def classify_points(self, v_c: Viewpoint) -> None:
        """Run classification to completion immediately."""
        for _ in self.classify_points_step_by_step(v_c):
            pass

    # def classify_points(self, v_c: Viewpoint) -> None:
    #     """
    #     Algorithm 3: ClassifyPoints(v_c, P', P_c, P_f, P_o, X_cap)
    #     """
    #     x_c = v_c.position.copy()
    #     eps = self.gamma_see.epsilon
    #     r = self.gamma_see.r_neighbor
    #     k_min = self.gamma_see.k_min

    #     # Line 2: R <- empty
    #     R: Set[int] = set()

    #     # Line 3: for all p in P' do
    #     for p in self.P_prime:
    #         # Line 4: if Neighbours_r(P, eps, p) == empty then
    #         eps_neighbors = self._find_neighbor_indices(p, eps)
    #         if len(eps_neighbors) == 0:
    #             # Line 5: P <- P U {p}
    #             new_idx = len(self.P_coords)
    #             self.P_coords.append(p.copy())

    #             # Line 6: Q <- {p} U Neighbours_r(P, r, p)
    #             r_neighbors = self._find_neighbor_indices(p, r)
    #             # Use a deque for FIFO queue processing
    #             Q_set: Set[int] = set(r_neighbors)
    #             Q_set.add(new_idx)
    #             Q: deque[int] = deque(Q_set)

    #             # Line 7: X_cap[p] <- x_c
    #             self.X_cap[new_idx] = x_c.copy()

    #             # Line 8: for all q in Q do
    #             while Q:
    #                 q = Q.popleft()

    #                 # Line 9: if q not in P_c then
    #                 if q not in self.P_c_idx:
    #                     # Line 10: N_q <- Neighbours_r(P, r, q)
    #                     N_q = set(self._find_neighbor_indices(self.P_coords[q], r))

    #                     # Line 11: if |N_q| < k_min then
    #                     if len(N_q) < k_min:
    #                         # Line 12: if N_q intersect P_c != empty then
    #                         if len(N_q.intersection(self.P_c_idx)) > 0:
    #                             # Line 13: P_f <- P_f U {q}
    #                             self.P_f_idx.add(q)
    #                             # Lines 14-16: if q in P_o then P_o <- P_o \ {q}
    #                             self.P_o_idx.discard(q)
    #                         else:
    #                             # Line 18: P_o <- P_o U {q}
    #                             self.P_o_idx.add(q)
    #                     else:
    #                         # Line 20-21: else P_c <- P_c U {q}
    #                         self.P_c_idx.add(q)
    #                         # Lines 22-24: if q in P_f then P_f <- P_f \ {q}
    #                         self.P_f_idx.discard(q)
    #                         # Lines 25-27: if q in P_o then P_o <- P_o \ {q}
    #                         self.P_o_idx.discard(q)

    #                         # Line 28: if p != q and q not in R then
    #                         if q != new_idx and q not in R:
    #                             # Line 29: Q <- Q U N_q
    #                             for n_idx in N_q:
    #                                 if n_idx not in Q_set:
    #                                     Q_set.add(n_idx)
    #                                     Q.append(n_idx)
    #                             # Line 30: R <- R U {q}
    #                             R.add(q)

    def adjust_view(self, v_c: Viewpoint) -> None:
        """Line 9: AdjustView(v_c, f_c, P', P_f, P_o, V, X_cap, E_surf)"""
        # Placeholder: modify orientation or position to maintain visibility of f_c
        pass

    # -------------------------------------------------------------------------
    # Algorithm 6: DirectNormal(v_c, f, e_n, P')
    # -------------------------------------------------------------------------
    def direct_normal(self, v_c: Viewpoint, f: np.ndarray, e_n: np.ndarray) -> np.ndarray:
        """
        Algorithm 6: Resolves the sign ambiguity of e_n[cite: 5].
        """
        x_c = v_c.position  # Line 1[cite: 5]
        v = self.gamma_see.v_step
        w_plus = (f - x_c).copy()   # Line 2[cite: 5]
        w_minus = (f - x_c).copy()  # Line 3[cite: 5]

        if not self.P_prime:
            # Fallback: face toward the current viewpoint
            to_cam = x_c - f
            return e_n if np.dot(e_n, to_cam) >= 0.0 else -e_n

        P_prime_arr = np.array(self.P_prime)
        dists_to_xc = np.linalg.norm(P_prime_arr - x_c, axis=1)

        iters = 0
        while iters < self.gamma_see.max_direct_iters:  # Line 4: repeat[cite: 5]
            iters += 1
            w_plus += v * e_n   # Line 5[cite: 5]
            w_minus -= v * e_n  # Line 6[cite: 5]

            norm_wp = float(np.linalg.norm(w_plus))
            norm_wm = float(np.linalg.norm(w_minus))
            if norm_wp < 1e-6 or norm_wm < 1e-6:
                break

            # Line 7: J subset of P' closer than ||w+||[cite: 5]
            mask = dists_to_xc < norm_wp
            if not np.any(mask):
                break
            J = (P_prime_arr[mask] - x_c) / dists_to_xc[mask, np.newaxis]

            # Line 8-9: Directional neighborhood check with tolerance v[cite: 5]
            u_plus = w_plus / norm_wp
            u_minus = w_minus / norm_wm

            dists_p = np.linalg.norm(J - u_plus, axis=1)
            dists_m = np.linalg.norm(J - u_minus, axis=1)

            n_w_plus_count = int(np.sum(dists_p < v))
            n_w_minus_count = int(np.sum(dists_m < v))

            # Line 10: until N_w+ == empty or N_w- == empty[cite: 5]
            if n_w_plus_count == 0 or n_w_minus_count == 0:
                # Line 11-13[cite: 5]
                if n_w_plus_count != 0 and n_w_minus_count == 0:
                    e_n = -e_n  # Line 12[cite: 5]
                break

        # Safety: verify e_n points towards free space / camera
        to_cam = x_c - f
        if np.dot(e_n, to_cam) < 0.0:
            e_n = -e_n

        return e_n

    # -------------------------------------------------------------------------
    # Algorithm 5: EstimateSurface(v_c, f, P, P', E_surf)
    # -------------------------------------------------------------------------
    def estimate_surface(self, v_c: Viewpoint, f_idx: int) -> SurfaceFrame2D:
        """
        Algorithm 5: Computes e_n, e_f, e_b at frontier point f in 2D[cite: 4].
        """
        f = self.P_coords[f_idx]
        r = self.gamma_see.r_neighbor

        # Line 1: N_f <- Neighbours_r(P, r, f) U {f}[cite: 4]
        n_indices = self._find_neighbor_indices(f, r)
        pts = [self.P_coords[i] for i in n_indices] + [f]  # shape (K, 2)
        pts_arr = np.array(pts)

        # Line 2: D <- [p1 - f, ..., pn - f][cite: 4]
        D = (pts_arr - f).T  # Shape (2, K)

        # Line 3: A <- D * D^T[cite: 4]
        A = D @ D.T  # Shape (2, 2)

        # Line 4-5: Eigenvalues & Eigenvectors of A[cite: 4]
        eigenvals, eigenvecs = np.linalg.eigh(A)  # Sorted in ascending order

        # Line 6: e_n <- eigenvector for min eigenvalue[cite: 4]
        e_n = eigenvecs[:, 0]
        norm_en = np.linalg.norm(e_n)
        e_n = e_n / norm_en if norm_en > 1e-6 else np.array([1.0, 0.0])

        # Line 7: DirectNormal(v_c, f, e_n, P')[cite: 4]
        e_n = self.direct_normal(v_c, f, e_n)

        # Line 8: p_bar <- (1 / |N_f|) * sum(f - p)[cite: 4]
        p_bar = np.mean(f - pts_arr, axis=0)

        # Line 9-10: e_f candidate and sign alignment with p_bar[cite: 4]
        e_f = np.array([-e_n[1], e_n[0]])
        if np.dot(p_bar, e_f) < 0.0:
            e_f = -e_f

        # Line 11-12: e_b = 1.0 (pointing out of screen); store frame[cite: 4]
        frame = SurfaceFrame2D(e_n=e_n, e_f=e_f, e_b=1.0)
        self.E_surf[f_idx] = frame
        return frame

    # -------------------------------------------------------------------------
    # Algorithm 4: ProposeViews(v_c, P, P', P_f, V, E_surf)
    # -------------------------------------------------------------------------
    def propose_views(self, v_c: Viewpoint) -> None:
        """
        Algorithm 4: For each unassigned frontier, estimate surface and propose V[f].
        """
        d = self.gamma_see.d_standoff

        # Line 2: for all f in P_f and V[f] == NULL do[cite: 3]
        for f_idx in list(self.P_f_idx):
            if f_idx not in self.V_map:
                f = self.P_coords[f_idx]

                # Line 3: EstimateSurface(v_c, f, P, P', E_surf)[cite: 3]
                frame = self.estimate_surface(v_c, f_idx)

                # Line 4: V[f] <- (f + d * e_n, -e_n)[cite: 3]
                cand_pos = f + d * frame.e_n
                cand_heading = float(np.arctan2(-frame.e_n[1], -frame.e_n[0]))

                self.V_map[f_idx] = Viewpoint(
                    position=cand_pos,
                    heading=cand_heading,
                    frontier_idx=f_idx,
                )

    def refine_views(self, v_c: Viewpoint) -> None:
        """Line 12: RefineViews(v_c, P, P_f, P_o, V, X_cap, E_surf)"""
        # Placeholder: filter collisions and evaluate information gain / coverage
        pass

    def graph_views(self, v_c: Viewpoint) -> None:
        """Line 13: GraphViews(v_c, P_f, V, G)"""
        # Placeholder: construct/update roadmap graph connecting candidates in V
        pass

    def select_nbv(self, v_c: Viewpoint) -> Optional[Viewpoint]:
        """
        Line 14: SelectNBV(v_c, f_c, G)
        Basic filler: uniformly samples one candidate viewpoint from V[cite: 1].
        """
        if not self.V_map:
            return None

        # Sample randomly from current proposed viewpoints V[cite: 3]
        available_views = list(self.V_map.values())
        selected_view = np.random.choice(available_views)

        # Set the current focus frontier f_c corresponding to the chosen viewpoint[cite: 1, 3]
        if selected_view.frontier_idx is not None and selected_view.frontier_idx < len(self.P_coords):
            self.f_c = self.P_coords[selected_view.frontier_idx].copy()

        return selected_view

    # -------------------------------------------------------------------------
    # Integration with Simulation Loop
    # -------------------------------------------------------------------------

    def select_next_waypoint(
        self,
        robot: Robot,
        environment: Environment,
        observations: List[RayHit],
    ) -> Optional[np.ndarray]:
        """
        Executes one pass of the SEE repeat loop (lines 5-15) and returns
        the next target [x, y] waypoint coordinate.
        """
        if self.is_complete:
            return None

        # Current robot configuration v_c
        v_c = Viewpoint(position=robot.state.position.copy(), heading=robot.state.theta)

        # Line 6: GetNewMeasurements(v_c, P')
        self.get_new_measurements(v_c, observations)

        # Line 7: ClassifyPoints(v_c, P', P_c, P_f, P_o, X_cap)
        self.classify_points(v_c)

        # Lines 8-10: if f_c in P_f then AdjustView(...)
        if self.f_c is not None and any(np.allclose(self.f_c, pt) for pt in self.P_f):
            self.adjust_view(v_c)

        # Line 11: ProposeViews(v_c, P, P', P_f, V, E_surf)
        self.propose_views(v_c)

        # Line 12: RefineViews(v_c, P, P_f, P_o, V, X_cap, E_surf)
        self.refine_views(v_c)

        # Line 13: GraphViews(v_c, P_f, V, G)
        self.graph_views(v_c)

        # Line 14: SelectNBV(v_c, f_c, G)
        best_view = self.select_nbv(v_c)

        # Line 15: until P_f = empty
        # (Once classification is active, empty frontiers signal completion)
        if len(self.P_f) == 0 and len(self.P_c) > 0:
            self.is_complete = True
            return None

        if best_view is not None:
            margin = robot.config.radius + 0.1
            target_x = float(np.clip(best_view.position[0], environment.x_min + margin, environment.x_max - margin))
            target_y = float(np.clip(best_view.position[1], environment.y_min + margin, environment.y_max - margin))

            return Viewpoint(
                position=np.array([target_x, target_y]),
                heading=best_view.heading,
                frontier_idx=best_view.frontier_idx,
            )

        return None

    # -------------------------------------------
    # --- Visual helpers ------------------------
    # -------------------------------------------
    def start_detailed_classification(self, robot_state: RobotState, observations: List[RayHit]) -> None:
        """Initializes step-by-step classification generator."""
        v_c = Viewpoint(position=robot_state.position.copy(), heading=robot_state.theta)
        self.get_new_measurements(v_c, observations)
        self.active_classifier_gen = self.classify_points_step_by_step(v_c)
        self.current_debug_state = None

    def step_detailed_classification(self) -> bool:
        """Advance one classification step. Returns True if more steps remain, False when finished."""
        if self.active_classifier_gen is None:
            return False
        try:
            self.current_debug_state = next(self.active_classifier_gen)
            return True
        except StopIteration:
            self.active_classifier_gen = None
            self.current_debug_state = None
            return False