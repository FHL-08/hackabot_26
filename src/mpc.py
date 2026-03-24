from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import numpy as np


PHASE_GATHER = 1
PHASE_PUSH = 2


@dataclass
class ControllerParams:
    t0: float = 0.0
    tf: float = 50.0
    dt: float = 0.03
    N: int = 10

    # Robot and geometry parameters
    r: float = 14.5 / 1000.0
    b: float = 0.04
    D: float = 0.02
    robot_radius: float = 0.04
    u_max: float = 0.40
    u_ca_max: float = 0.25
    du_max: float = 0.55
    u_filter_alpha: float = 0.30
    payload_theta_rate_max: float = np.pi / 4.0
    payload_theta_filter_alpha: float = 0.35

    # Collision / avoidance settings
    obstacle_radius: float = 0.1125
    d_safe_obs: float = obstacle_radius + robot_radius + 0.01
    d_safe_rr: float = 2.0 * robot_radius + 0.01
    rho_obs: float = 0.15
    rho_rr: float = 0.15
    k_obs: float = 0.08
    k_rr: float = 0.025
    k_obs_gather: float = 0.14
    rho_obs_gather: float = 0.22
    obs_mag_max_gather: float = 0.14
    k_obs_push: float = 0.12
    rho_obs_push: float = 0.20
    obs_mag_max_push: float = 0.12
    k_rr_separating_scale: float = 0.25
    rr_activation_power: float = 2.0
    rr_pair_mag_max: float = 0.04
    rr_trigger_clearance: float = 0.015
    gather_ca_scale_min: float = 0.20
    gather_ca_err_ref: float = 0.18
    obs_activation_power: float = 2.0
    obs_mag_max: float = 0.08
    payload_rep_max: float = 0.22
    payload_clearance_margin: float = 0.03
    push_obs_boost_dist: float = 0.06
    push_obs_boost_gain: float = 1.1
    payload_obs_weights: Tuple[float, float, float] = (1.0, 1.0, 1.25)
    payload_hard_guard_dist: float = 0.05
    payload_hard_guard_gain: float = 0.20
    eps_dist: float = 1e-3

    # Gather phase convergence shaping
    u_gather_max: float = 0.22
    u_gather_min: float = 0.04
    gather_err_speed_ref: float = 0.14
    gather_switch_vel_tol: float = 0.035
    phase_blend_time: float = 0.75

    # Switching and hold
    gather_tol: float = 0.06
    min_gather_hold_s: float = 0.9

    # Challenge runtime budget
    challenge_time_limit_s: float = 240.0


@dataclass
class APFParams:
    k_att: float = 5.0
    k_rep: float = 0.95
    rho_payload: float = 1.15
    alpha_payload: float = 0.015


@dataclass
class StepOutput:
    wheel_rates_lr: np.ndarray
    wheel_rates_lr_flat: np.ndarray
    phase: int
    diagnostics: Dict[str, float]


class TwoPhaseRealtimeController:
    """Realtime controller ported from MATLAB two-phase gather + push logic.

    The controller keeps the virtual payload state internally and outputs wheel
    angular velocities for 3 differential-drive robots at every call to step().
    """

    def __init__(
        self,
        params: ControllerParams,
        apf_params: APFParams,
        obstacle_array: np.ndarray,
        payload_center_init: np.ndarray,
        payload_goal: np.ndarray,
        payload_radius: float,
    ) -> None:
        self.params = params
        self.apf = apf_params
        self.obstacle_array = np.asarray(
            obstacle_array, dtype=np.float64).reshape(2, -1)
        self.payload_goal = np.asarray(
            payload_goal, dtype=np.float64).reshape(2)
        self.payload_center = np.asarray(
            payload_center_init, dtype=np.float64).reshape(2).copy()
        self.payload_theta = 0.0
        self.d_safe_payload_obs = (
            self.params.obstacle_radius
            + float(payload_radius)
            + self.params.payload_clearance_margin
        )

        contact_radius = payload_radius + self.params.robot_radius
        self.contact_offsets_body = np.array(
            [
                [0.0, contact_radius *
                    np.sin(np.deg2rad(60.0)), -contact_radius * np.sin(np.deg2rad(60.0))],
                [-contact_radius, -contact_radius *
                    np.cos(np.deg2rad(60.0)), -contact_radius * np.cos(np.deg2rad(60.0))],
            ],
            dtype=np.float64,
        )

        self.phase = PHASE_GATHER
        self.step_idx = 0
        self.gather_counter = 0
        self.phase_switch_step = -1
        self.gather_contact_switch = np.zeros((2, 3), dtype=np.float64)

        self.min_gather_hold_steps = int(
            round(self.params.min_gather_hold_s / self.params.dt))
        self.phase_blend_steps = max(
            1, int(round(self.params.phase_blend_time / self.params.dt)))

        self.u_prev = np.zeros((2, 3), dtype=np.float64)

        self.max_total_steps = max(
            1, int(np.floor(self.params.challenge_time_limit_s / self.params.dt)))
        self.push_plan_contact_world: np.ndarray | None = None
        self.push_plan_centers: np.ndarray | None = None
        self.push_plan_thetas: np.ndarray | None = None
        self.push_plan_len = 0
        self.push_plan_cursor = 0

        self._build_mpc_matrices()

    def _build_mpc_matrices(self) -> None:
        p = self.params

        q_block = np.diag(np.tile(np.array([50.0, 50.0]), 3))
        s_block = np.diag(np.tile(np.array([6.0, 6.0]), 3))

        q_bar = np.kron(np.eye(p.N + 1), q_block)
        s_bar = np.kron(np.eye(p.N), s_block)

        L = np.tril(np.ones((p.N, p.N), dtype=np.float64))
        gamma_left = np.vstack((np.zeros((1, p.N), dtype=np.float64), L))
        gamma = p.dt * np.kron(gamma_left, np.eye(6))

        h = 2.0 * (gamma.T @ q_bar @ gamma + s_bar)
        self.gq = gamma.T @ q_bar

        # H is positive definite in this setup; store Cholesky factor for fast solves.
        self.h_chol = np.linalg.cholesky(h)

    def reset(
        self,
        payload_center: np.ndarray,
        payload_theta: float = 0.0,
    ) -> None:
        self.payload_center = np.asarray(
            payload_center, dtype=np.float64).reshape(2).copy()
        self.payload_theta = float(payload_theta)
        self.phase = PHASE_GATHER
        self.step_idx = 0
        self.gather_counter = 0
        self.phase_switch_step = -1
        self.gather_contact_switch.fill(0.0)
        self.u_prev.fill(0.0)
        self.push_plan_contact_world = None
        self.push_plan_centers = None
        self.push_plan_thetas = None
        self.push_plan_len = 0
        self.push_plan_cursor = 0

    def _payload_apf_step(
        self,
        payload_center: np.ndarray,
        payload_theta: float,
    ) -> Tuple[np.ndarray, float, float]:
        p = self.params
        p_curr = payload_center
        f_att = self.apf.k_att * (self.payload_goal - p_curr)

        d_payload_clear_all = (
            np.linalg.norm(self.obstacle_array - p_curr[:, None], axis=0)
            - self.d_safe_payload_obs
        )
        min_payload_obs_dist = float(np.min(d_payload_clear_all))

        f_rep = np.zeros(2, dtype=np.float64)
        for o in range(self.obstacle_array.shape[1]):
            obs_vec = p_curr - self.obstacle_array[:, o]
            d_center = float(np.linalg.norm(obs_vec))
            d_clear_payload = d_center - self.d_safe_payload_obs

            if d_clear_payload < self.apf.rho_payload:
                d_eff = max(d_clear_payload, p.eps_dist)
                grad = obs_vec / max(d_center, p.eps_dist)

                activation = (self.apf.rho_payload - d_clear_payload) / max(
                    self.apf.rho_payload, p.eps_dist
                )
                activation = np.clip(
                    activation, 0.0, 1.0) ** p.obs_activation_power

                boost = 1.0
                if d_clear_payload < p.push_obs_boost_dist:
                    boost = 1.0 + p.push_obs_boost_gain * (
                        (p.push_obs_boost_dist - d_clear_payload)
                        / max(p.push_obs_boost_dist, p.eps_dist)
                    )

                obs_weight = p.payload_obs_weights[min(
                    o, len(p.payload_obs_weights) - 1)]
                mag_rep = (
                    obs_weight
                    * self.apf.k_rep
                    * activation
                    * boost
                    * (1.0 / d_eff - 1.0 / self.apf.rho_payload)
                    / (d_eff ** 2)
                )
                f_rep += mag_rep * grad

        f_rep = clip_vec(f_rep, p.payload_rep_max)

        f_total = self.apf.alpha_payload * (f_att + f_rep)

        # Hard guard: near obstacles, remove inward payload velocity components.
        for o in range(self.obstacle_array.shape[1]):
            obs_vec = p_curr - self.obstacle_array[:, o]
            d_center = float(np.linalg.norm(obs_vec))
            d_clear_payload = d_center - self.d_safe_payload_obs

            if d_clear_payload < p.payload_hard_guard_dist:
                n_out = obs_vec / max(d_center, p.eps_dist)
                v_in = float(np.dot(f_total, n_out))

                if v_in < 0.0:
                    f_total = f_total - v_in * n_out

                if d_clear_payload < 0.0:
                    guard_activation = min(
                        (-d_clear_payload) /
                        max(p.payload_hard_guard_dist, p.eps_dist),
                        1.0,
                    ) ** 2
                    obs_weight = p.payload_obs_weights[min(
                        o, len(p.payload_obs_weights) - 1)]
                    f_total = f_total + obs_weight * \
                        p.payload_hard_guard_gain * guard_activation * n_out

        payload_center_next = p_curr + f_total * p.dt

        payload_theta_next = payload_theta
        if np.linalg.norm(f_total) > 1e-3:
            theta_des = np.arctan2(f_total[1], f_total[0]) - np.pi / 2.0
            dtheta = wrap_to_pi(theta_des - payload_theta)
            dtheta_max = p.payload_theta_rate_max * p.dt
            dtheta = np.clip(dtheta, -dtheta_max, dtheta_max)
            theta_rate_limited = payload_theta + dtheta
            payload_theta_next = wrap_to_pi(
                (1.0 - p.payload_theta_filter_alpha) * payload_theta
                + p.payload_theta_filter_alpha * theta_rate_limited
            )

        return payload_center_next, payload_theta_next, min_payload_obs_dist

    def _generate_push_plan(self, remaining_steps: int) -> None:
        # Precompute APF payload contacts once at phase switch for deterministic runtime.
        rem = max(1, int(remaining_steps))
        centers = np.zeros((2, rem + 1), dtype=np.float64)
        thetas = np.zeros(rem + 1, dtype=np.float64)
        contacts = np.zeros((2, 3, rem + 1), dtype=np.float64)

        centers[:, 0] = self.payload_center
        thetas[0] = self.payload_theta
        contacts[:, :, 0] = self.payload_center[:, None] + \
            rot2(self.payload_theta) @ self.contact_offsets_body

        for k in range(1, rem + 1):
            centers[:, k], thetas[k], _ = self._payload_apf_step(
                centers[:, k - 1], thetas[k - 1])
            contacts[:, :, k] = centers[:, k, None] + \
                rot2(thetas[k]) @ self.contact_offsets_body

        self.push_plan_contact_world = contacts
        self.push_plan_centers = centers
        self.push_plan_thetas = thetas
        self.push_plan_len = rem + 1
        self.push_plan_cursor = 0

    def step(self, robot_positions: np.ndarray, robot_headings: np.ndarray) -> StepOutput:
        """Compute wheel rates from current measured robot states.

        Args:
            robot_positions: shape (2, 3), columns are [p1 p2 p3].
            robot_headings: shape (3,), headings [theta1, theta2, theta3].
        """
        p = self.params
        robot_positions = np.asarray(
            robot_positions, dtype=np.float64).reshape(2, 3)
        robot_headings = np.asarray(
            robot_headings, dtype=np.float64).reshape(3)

        if self.step_idx >= self.max_total_steps:
            wheel_rates_lr = np.zeros((2, 3), dtype=np.float64)
            return StepOutput(
                wheel_rates_lr=wheel_rates_lr,
                wheel_rates_lr_flat=np.zeros(6, dtype=np.float64),
                phase=self.phase,
                diagnostics={
                    "time_expired": 1.0,
                    "mission_time_s": float(self.step_idx * p.dt),
                    "mission_time_remaining_s": 0.0,
                },
            )

        p_stack = robot_positions.T.reshape(6)
        hat_p = np.tile(p_stack, p.N + 1)

        gather_error = 0.0
        min_payload_obs_dist = float("inf")

        if self.phase == PHASE_GATHER:
            r_payload = rot2(self.payload_theta)
            contact_world = self.payload_center[:, None] + \
                r_payload @ self.contact_offsets_body

            err_mat = robot_positions - contact_world
            gather_error = float(np.linalg.norm(err_mat.T.reshape(6)))

            gather_err_i = np.linalg.norm(err_mat, axis=0)
            max_gather_err = float(np.max(gather_err_i))
            max_body_speed = float(np.max(np.linalg.norm(self.u_prev, axis=0)))
            min_payload_obs_dist = float(
                np.min(np.linalg.norm(self.obstacle_array -
                       self.payload_center[:, None], axis=0) - self.d_safe_payload_obs)
            )

            if (max_gather_err < p.gather_tol) and (max_body_speed < p.gather_switch_vel_tol):
                self.gather_counter += 1
            else:
                self.gather_counter = 0

            if self.gather_counter >= self.min_gather_hold_steps:
                self.phase = PHASE_PUSH
                self.phase_switch_step = self.step_idx
                self.gather_contact_switch = contact_world.copy()
                remaining = self.max_total_steps - self.step_idx
                self._generate_push_plan(remaining_steps=remaining)
        else:
            if (
                (self.push_plan_contact_world is None)
                or (self.push_plan_centers is None)
                or (self.push_plan_thetas is None)
            ):
                remaining = self.max_total_steps - self.step_idx
                self._generate_push_plan(remaining_steps=remaining)

            assert self.push_plan_contact_world is not None
            assert self.push_plan_centers is not None
            assert self.push_plan_thetas is not None

            self.push_plan_cursor = min(self.push_plan_cursor + 1,
                                        self.push_plan_len - 1)
            contact_world_push = self.push_plan_contact_world[:,
                                                              :, self.push_plan_cursor]

            if (
                (self.phase_switch_step >= 0)
                and (self.step_idx <= (self.phase_switch_step + self.phase_blend_steps))
            ):
                beta = (self.step_idx - self.phase_switch_step) / \
                    self.phase_blend_steps
                beta = float(np.clip(beta, 0.0, 1.0))
                contact_world = (
                    1.0 - beta) * self.gather_contact_switch + beta * contact_world_push
            else:
                contact_world = contact_world_push

            self.payload_center = self.push_plan_centers[:, self.push_plan_cursor].copy(
            )
            self.payload_theta = float(
                self.push_plan_thetas[self.push_plan_cursor])
            min_payload_obs_dist = float(
                np.min(np.linalg.norm(self.obstacle_array -
                       self.payload_center[:, None], axis=0) - self.d_safe_payload_obs)
            )

        p_ref_now = contact_world.T.reshape(6)
        p_ref_stack = np.tile(p_ref_now, p.N + 1)

        lambda_j = hat_p - p_ref_stack
        f_j = 2.0 * (self.gq @ lambda_j)

        y = np.linalg.solve(self.h_chol, -f_j)
        opt_u = np.linalg.solve(self.h_chol.T, y)
        u_matrix = opt_u.reshape((6, p.N), order="F")

        # Per-robot speed cap over the horizon
        cap_horizon_rows(u_matrix[0:2, :], p.u_max, p.eps_dist)
        cap_horizon_rows(u_matrix[2:4, :], p.u_max, p.eps_dist)
        cap_horizon_rows(u_matrix[4:6, :], p.u_max, p.eps_dist)

        if self.phase == PHASE_GATHER:
            err1 = float(np.linalg.norm(
                robot_positions[:, 0] - contact_world[:, 0]))
            err2 = float(np.linalg.norm(
                robot_positions[:, 1] - contact_world[:, 1]))
            err3 = float(np.linalg.norm(
                robot_positions[:, 2] - contact_world[:, 2]))

            ug1 = p.u_gather_min + \
                (p.u_gather_max - p.u_gather_min) * \
                np.tanh(err1 / p.gather_err_speed_ref)
            ug2 = p.u_gather_min + \
                (p.u_gather_max - p.u_gather_min) * \
                np.tanh(err2 / p.gather_err_speed_ref)
            ug3 = p.u_gather_min + \
                (p.u_gather_max - p.u_gather_min) * \
                np.tanh(err3 / p.gather_err_speed_ref)

            cap_horizon_rows(u_matrix[0:2, :], ug1, p.eps_dist)
            cap_horizon_rows(u_matrix[2:4, :], ug2, p.eps_dist)
            cap_horizon_rows(u_matrix[4:6, :], ug3, p.eps_dist)

        u_nom = u_matrix[:, 0]

        u_ca, min_obs_dist, min_rr_dist = collision_avoidance_velocity(
            robot_positions,
            self.obstacle_array,
            p,
            u_nom,
            self.phase,
            gather_error,
        )

        u_ca[:, 0] = clip_vec(u_ca[:, 0], p.u_ca_max)
        u_ca[:, 1] = clip_vec(u_ca[:, 1], p.u_ca_max)
        u_ca[:, 2] = clip_vec(u_ca[:, 2], p.u_ca_max)

        u_total = u_nom.reshape(3, 2).T + u_ca

        u_des = np.zeros((2, 3), dtype=np.float64)
        u_rate = np.zeros((2, 3), dtype=np.float64)
        u_now = np.zeros((2, 3), dtype=np.float64)

        for i in range(3):
            u_des[:, i] = smooth_clip_vec(u_total[:, i], p.u_max)
            u_rate[:, i] = rate_limit_vec(
                u_des[:, i], self.u_prev[:, i], p.du_max, p.dt)
            u_now[:, i] = (1.0 - p.u_filter_alpha) * \
                self.u_prev[:, i] + p.u_filter_alpha * u_rate[:, i]
            u_now[:, i] = clip_vec(u_now[:, i], p.u_max)

        self.u_prev = u_now.copy()

        wheel_rates_rl = np.zeros((2, 3), dtype=np.float64)
        for i in range(3):
            phi_i, theta_i = wheel_and_heading_update(
                robot_headings[i], u_now[:, i], p)
            wheel_rates_rl[:, i] = phi_i
            robot_headings[i] = theta_i

        # Convert to user-facing left-right order.
        wheel_rates_lr = np.vstack(
            (wheel_rates_rl[1, :], wheel_rates_rl[0, :]))
        wheel_rates_lr_flat = wheel_rates_lr.T.reshape(6)

        diagnostics = {
            "min_obs_dist": float(min_obs_dist),
            "min_rr_dist": float(min_rr_dist),
            "min_payload_obs_dist": float(min_payload_obs_dist),
            "u1_norm": float(np.linalg.norm(u_now[:, 0])),
            "u2_norm": float(np.linalg.norm(u_now[:, 1])),
            "u3_norm": float(np.linalg.norm(u_now[:, 2])),
            "gather_error": float(gather_error),
            "time_expired": 0.0,
            "mission_time_s": float(self.step_idx * p.dt),
            "mission_time_remaining_s": float(
                max(0.0, (self.max_total_steps - self.step_idx) * p.dt)),
        }

        self.step_idx += 1

        return StepOutput(
            wheel_rates_lr=wheel_rates_lr,
            wheel_rates_lr_flat=wheel_rates_lr_flat,
            phase=self.phase,
            diagnostics=diagnostics,
        )

    def step_wheel_rates(
        self,
        robot_positions: np.ndarray,
        robot_headings: np.ndarray,
    ) -> np.ndarray:
        """Return only wheel rates in [L1, R1, L2, R2, L3, R3] order."""
        return self.step(robot_positions, robot_headings).wheel_rates_lr_flat


def collision_avoidance_velocity(
    robot_positions: np.ndarray,
    obstacle_array: np.ndarray,
    params: ControllerParams,
    u_nom: np.ndarray,
    phase: int,
    gather_err: float,
) -> Tuple[np.ndarray, float, float]:
    p_all = robot_positions
    u_nom_all = u_nom.reshape(3, 2).T
    u_ca = np.zeros((2, 3), dtype=np.float64)

    min_obs_dist = np.inf
    min_rr_dist = np.inf

    for i in range(3):
        ui = np.zeros(2, dtype=np.float64)
        pi = p_all[:, i]

        for o in range(obstacle_array.shape[1]):
            d_center = float(np.linalg.norm(pi - obstacle_array[:, o]))
            d_clear = d_center - params.d_safe_obs
            min_obs_dist = min(min_obs_dist, d_clear)

            if phase == PHASE_GATHER:
                rho_obs_eff = params.rho_obs_gather
                k_obs_eff = params.k_obs_gather
                obs_mag_max_eff = params.obs_mag_max_gather
            else:
                rho_obs_eff = params.rho_obs_push
                k_obs_eff = params.k_obs_push
                obs_mag_max_eff = params.obs_mag_max_push

            if d_clear < rho_obs_eff:
                grad = (pi - obstacle_array[:, o]) / \
                    max(d_center, params.eps_dist)
                activation = (rho_obs_eff - d_clear) / \
                    max(rho_obs_eff, params.eps_dist)
                activation = np.clip(
                    activation, 0.0, 1.0) ** params.obs_activation_power
                mag_obs = min(k_obs_eff * activation, obs_mag_max_eff)
                ui += mag_obs * grad

        u_ca[:, i] += ui

    pairs = ((0, 1), (0, 2), (1, 2))

    if phase == PHASE_GATHER:
        rr_phase_scale = max(params.gather_ca_scale_min,
                             np.exp(-gather_err / params.gather_ca_err_ref))
    else:
        rr_phase_scale = 1.0

    for i, j in pairs:
        pi = p_all[:, i]
        pj = p_all[:, j]

        d_center = float(np.linalg.norm(pi - pj))
        d_clear = d_center - params.d_safe_rr
        min_rr_dist = min(min_rr_dist, d_clear)

        if d_clear < params.rr_trigger_clearance:
            dir_ij = (pi - pj) / max(d_center, params.eps_dist)

            activation = (params.rr_trigger_clearance - d_clear) / \
                max(params.rr_trigger_clearance, params.eps_dist)
            activation = np.clip(
                activation, 0.0, 1.0) ** params.rr_activation_power

            rel_speed = float(
                np.dot(u_nom_all[:, i] - u_nom_all[:, j], dir_ij))
            closing_scale = params.k_rr_separating_scale if rel_speed > 0.0 else 1.0

            penetration = max(-d_clear, 0.0)
            mag_raw = params.k_rr * \
                (activation + (penetration / max(params.d_safe_rr, params.eps_dist)) ** 2)
            mag = min(rr_phase_scale * closing_scale *
                      mag_raw, params.rr_pair_mag_max)

            u_ca[:, i] += mag * dir_ij
            u_ca[:, j] -= mag * dir_ij

    return u_ca, float(min_obs_dist), float(min_rr_dist)


def clip_vec(v: np.ndarray, vmax: float) -> np.ndarray:
    nv = float(np.linalg.norm(v))
    if nv > vmax:
        return (v / nv) * vmax
    return v


def smooth_clip_vec(v: np.ndarray, vmax: float) -> np.ndarray:
    nv = float(np.linalg.norm(v))
    return v * (vmax / np.sqrt(nv * nv + vmax * vmax))


def rate_limit_vec(v_cmd: np.ndarray, v_prev: np.ndarray, amax: float, dt: float) -> np.ndarray:
    dv = v_cmd - v_prev
    dv_max = amax * dt
    ndv = float(np.linalg.norm(dv))
    if ndv > dv_max:
        return v_prev + (dv / ndv) * dv_max
    return v_cmd


def cap_horizon_rows(u_block: np.ndarray, vmax_block: float, eps_dist: float) -> None:
    n = np.linalg.norm(u_block, axis=0)
    s = np.minimum(1.0, vmax_block / np.maximum(n, eps_dist))
    u_block *= s


def wrap_to_pi(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def rot2(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def wheel_and_heading_update(
    theta_now: float,
    u: np.ndarray,
    params: ControllerParams,
) -> Tuple[np.ndarray, float]:
    inv_r_q = np.array(
        [[np.cos(theta_now), np.sin(theta_now)],
         [-np.sin(theta_now) / params.D, np.cos(theta_now) / params.D]],
        dtype=np.float64,
    )

    bar_m = np.array(
        [[params.b, params.b], [params.b, -params.b]], dtype=np.float64)
    r_t = np.array(
        [[np.cos(theta_now), -np.sin(theta_now)],
         [np.sin(theta_now), np.cos(theta_now)]],
        dtype=np.float64,
    )
    inv_s_q = (1.0 / (params.r * params.D)) * (bar_m @ r_t.T)

    v = inv_r_q @ u
    phi = inv_s_q @ u
    theta_next = theta_now + v[1] * params.dt
    return phi, float(theta_next)


def build_default_controller() -> TwoPhaseRealtimeController:
    """Factory with the same values as the provided MATLAB script."""
    params = ControllerParams()
    apf = APFParams()

    obstacle_array = np.array(
        [
            [0.22, -0.08, -0.08],
            [-0.16, -0.435, 0.22],
        ],
        dtype=np.float64,
    )
    payload_center_init = np.array([-0.34, -0.16], dtype=np.float64)
    payload_goal = np.array([0.26, 0.385], dtype=np.float64)
    payload_radius = 0.05

    return TwoPhaseRealtimeController(
        params=params,
        apf_params=apf,
        obstacle_array=obstacle_array,
        payload_center_init=payload_center_init,
        payload_goal=payload_goal,
        payload_radius=payload_radius,
    )


def cv_state_to_controller_input(
    cv_state: Mapping[str, Mapping[str, float]],
    robot_order: Iterable[str] = ("r1", "r2", "r3"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert CV state dict to controller arrays.

    Expected per-robot fields: x, y, theta.
    """
    order = tuple(robot_order)
    robot_positions = np.zeros((2, 3), dtype=np.float64)
    robot_headings = np.zeros(3, dtype=np.float64)

    for i, robot_id in enumerate(order):
        s = cv_state[robot_id]
        robot_positions[0, i] = float(s["x"])
        robot_positions[1, i] = float(s["y"])
        robot_headings[i] = float(s["theta"])

    return robot_positions, robot_headings


def wheel_vector_to_robot_packets(
    wheel_rates_lr_flat: np.ndarray,
    robot_order: Iterable[str] = ("r1", "r2", "r3"),
) -> Dict[str, np.ndarray]:
    """Map [L1,R1,L2,R2,L3,R3] to per-robot [left,right] vectors."""
    order = tuple(robot_order)
    v = np.asarray(wheel_rates_lr_flat, dtype=np.float64).reshape(6)
    return {
        order[0]: v[0:2],
        order[1]: v[2:4],
        order[2]: v[4:6],
    }


if __name__ == "__main__":
    # Example usage with CV-like input formatting.
    ctrl = build_default_controller()

    cv_state_sample = {
        "r1": {"x": -0.01, "y": -0.68, "theta": 0.70},
        "r2": {"x": 0.275, "y": -0.65, "theta": 1.2},
        "r3": {"x": -0.24, "y": -0.74, "theta": 1.5},
    }
    robot_pos, robot_theta = cv_state_to_controller_input(cv_state_sample)

    wheel_vec = ctrl.step_wheel_rates(robot_pos, robot_theta)
    per_robot_cmd = wheel_vector_to_robot_packets(wheel_vec)

    print("Wheel vector [L1,R1,L2,R2,L3,R3] [rad/s]:")
    print(wheel_vec)
    print("Per-robot vectors [left,right] [rad/s]:")
    print(per_robot_cmd)
