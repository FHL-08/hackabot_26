#!/usr/bin/env python3
"""Closed-loop MPC simulation with mock CV input and optional TCP send probe.

This script tests the exact runtime stack pieces you are using:
- Controller: mpc.build_default_controller()
- Input format: mock CV state -> cv_state_to_controller_input()
- Output: six wheel rates [L1,R1,L2,R2,L3,R3]
- Plant model: differential-drive kinematics consistent with MATLAB mapping
- Optional network path: broadcasts six-float command lines through BotCommandServer

Run examples:
  python mpc_mock_cv_simulation.py
  python mpc_mock_cv_simulation.py --tf 50 --network-probe
  python mpc_mock_cv_simulation.py --cv-noise-xy 0.003 --cv-noise-theta 0.03
"""

from __future__ import annotations

import argparse
import socket
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.laptop_server import BotCommandServer
from src.mpc import ControllerParams, build_default_controller, cv_state_to_controller_input

ROBOT_ORDER = ("r1", "r2", "r3")


@dataclass
class NetworkProbeStats:
    lines_sent: int
    lines_received_total: int
    per_client_received: Dict[int, int]


class NetworkProbe:
    """Loopback TCP probe for the same send path used by laptop_server auto mode."""

    def __init__(self, host: str, port: int, n_clients: int = 3) -> None:
        self.host = host
        self.port = int(port)
        self.n_clients = int(n_clients)
        self.server = BotCommandServer(
            host=self.host,
            port=self.port,
            debug=False,
            telem_on_change_only=True,
        )
        self.clients: list[socket.socket] = []
        self.buffers: list[bytes] = []
        self.lines_sent = 0
        self.per_client_received = {i: 0 for i in range(self.n_clients)}

    def start(self) -> None:
        self.server.start()
        time.sleep(0.10)

        for i in range(self.n_clients):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((self.host, self.port))
            sock.setblocking(False)
            self.clients.append(sock)
            self.buffers.append(b"")
            time.sleep(0.02)

    def stop(self) -> NetworkProbeStats:
        # Final drain before closing.
        self._drain_all_clients()

        for sock in self.clients:
            try:
                sock.close()
            except OSError:
                pass
        self.clients.clear()
        self.buffers.clear()

        self.server.stop()

        total_received = int(sum(self.per_client_received.values()))
        return NetworkProbeStats(
            lines_sent=int(self.lines_sent),
            lines_received_total=total_received,
            per_client_received=dict(self.per_client_received),
        )

    def broadcast_wheel_vector(self, wheel_lr_flat: np.ndarray) -> None:
        vals = np.asarray(wheel_lr_flat, dtype=np.float64).reshape(6)
        line = (
            f"{vals[0]:.6f} {vals[1]:.6f} "
            f"{vals[2]:.6f} {vals[3]:.6f} "
            f"{vals[4]:.6f} {vals[5]:.6f}"
        )
        self.server.broadcast_line(line)
        self.lines_sent += 1

        # Allow the socket thread to flush then count delivered lines.
        time.sleep(0.001)
        self._drain_all_clients()

    def _drain_all_clients(self) -> None:
        for idx, sock in enumerate(self.clients):
            while True:
                try:
                    chunk = sock.recv(4096)
                except BlockingIOError:
                    break
                except TimeoutError:
                    break
                except OSError:
                    break

                if not chunk:
                    break

                self.buffers[idx] += chunk
                while b"\n" in self.buffers[idx]:
                    _, rest = self.buffers[idx].split(b"\n", 1)
                    self.per_client_received[idx] += 1
                    self.buffers[idx] = rest


def default_initial_robot_state() -> Tuple[np.ndarray, np.ndarray]:
    positions = np.array(
        [
            [-0.01, 0.275, -0.24],
            [-0.68, -0.65, -0.74],
        ],
        dtype=np.float64,
    )
    headings = np.array([0.70, 1.2, 1.5], dtype=np.float64)
    return positions, headings


def make_mock_cv_state(
    robot_positions: np.ndarray,
    robot_headings: np.ndarray,
    noise_xy_std: float,
    noise_theta_std: float,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, float]]:
    cv_state: Dict[str, Dict[str, float]] = {}
    for i, robot_id in enumerate(ROBOT_ORDER):
        x = float(robot_positions[0, i] + rng.normal(0.0, noise_xy_std))
        y = float(robot_positions[1, i] + rng.normal(0.0, noise_xy_std))
        theta = float(robot_headings[i] + rng.normal(0.0, noise_theta_std))
        cv_state[robot_id] = {"x": x, "y": y, "theta": theta}
    return cv_state


def plant_step_from_wheel_lr(
    robot_positions: np.ndarray,
    robot_headings: np.ndarray,
    wheel_lr_flat: np.ndarray,
    params: ControllerParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance plant by one step using wheel-rate commands.

    Uses the same mapping structure as the MATLAB script:
    - convert wheel rates to body velocity via inverse of inv_S_q
    - heading update from inv_R_q * u
    - position update p_{k+1} = p_k + u*dt
    """
    next_positions = robot_positions.copy()
    next_headings = robot_headings.copy()
    body_u = np.zeros((2, 3), dtype=np.float64)

    wheel = np.asarray(wheel_lr_flat, dtype=np.float64).reshape(6)

    for i in range(3):
        left = float(wheel[2 * i])
        right = float(wheel[2 * i + 1])

        # Internal mapping used by controller code is right-left order.
        phi_rl = np.array([right, left], dtype=np.float64)

        theta_now = float(robot_headings[i])
        inv_r_q = np.array(
            [
                [np.cos(theta_now), np.sin(theta_now)],
                [-np.sin(theta_now) / params.D, np.cos(theta_now) / params.D],
            ],
            dtype=np.float64,
        )

        bar_m = np.array(
            [[params.b, params.b], [params.b, -params.b]], dtype=np.float64)
        r_t = np.array(
            [
                [np.cos(theta_now), -np.sin(theta_now)],
                [np.sin(theta_now), np.cos(theta_now)],
            ],
            dtype=np.float64,
        )
        inv_s_q = (1.0 / (params.r * params.D)) * (bar_m @ r_t.T)

        u = np.linalg.solve(inv_s_q, phi_rl)
        v = inv_r_q @ u

        next_headings[i] = theta_now + float(v[1]) * params.dt
        next_positions[:, i] = robot_positions[:, i] + u * params.dt
        body_u[:, i] = u

    return next_positions, next_headings, body_u


def run_simulation(args: argparse.Namespace) -> dict:
    controller = build_default_controller()
    params = controller.params

    robot_positions, robot_headings = default_initial_robot_state()

    steps = int(np.floor(float(args.tf) / params.dt))
    t = np.arange(steps + 1, dtype=np.float64) * params.dt

    p_hist = np.zeros((2, 3, steps + 1), dtype=np.float64)
    th_hist = np.zeros((3, steps + 1), dtype=np.float64)
    u_hist = np.zeros((2, 3, steps), dtype=np.float64)
    wheel_hist = np.zeros((6, steps), dtype=np.float64)
    payload_hist = np.zeros((2, steps + 1), dtype=np.float64)
    phase_hist = np.zeros(steps, dtype=np.int32)

    min_obs_hist = np.full(steps, np.nan, dtype=np.float64)
    min_rr_hist = np.full(steps, np.nan, dtype=np.float64)
    min_payload_obs_hist = np.full(steps, np.nan, dtype=np.float64)

    p_hist[:, :, 0] = robot_positions
    th_hist[:, 0] = robot_headings
    payload_hist[:, 0] = np.asarray(
        controller.payload_center, dtype=np.float64)

    rng = np.random.default_rng(args.seed)

    probe = None
    probe_stats = None
    if args.network_probe:
        probe = NetworkProbe(args.probe_host, args.probe_port, n_clients=3)
        probe.start()

    try:
        for k in range(steps):
            cv_state = make_mock_cv_state(
                robot_positions,
                robot_headings,
                noise_xy_std=args.cv_noise_xy,
                noise_theta_std=args.cv_noise_theta,
                rng=rng,
            )
            cv_pos, cv_theta = cv_state_to_controller_input(
                cv_state, robot_order=ROBOT_ORDER)

            step_out = controller.step(cv_pos, cv_theta)
            wheel_vec = np.asarray(
                step_out.wheel_rates_lr_flat, dtype=np.float64).reshape(6)

            if not np.all(np.isfinite(wheel_vec)):
                raise RuntimeError(
                    f"Non-finite wheel vector at step {k}: {wheel_vec}")

            if probe is not None:
                probe.broadcast_wheel_vector(wheel_vec)

            robot_positions, robot_headings, body_u = plant_step_from_wheel_lr(
                robot_positions,
                robot_headings,
                wheel_vec,
                params,
            )

            p_hist[:, :, k + 1] = robot_positions
            th_hist[:, k + 1] = robot_headings
            u_hist[:, :, k] = body_u
            wheel_hist[:, k] = wheel_vec
            payload_hist[:, k +
                         1] = np.asarray(controller.payload_center, dtype=np.float64)
            phase_hist[k] = int(step_out.phase)

            diag = step_out.diagnostics or {}
            min_obs_hist[k] = float(diag.get("min_obs_dist", np.nan))
            min_rr_hist[k] = float(diag.get("min_rr_dist", np.nan))
            min_payload_obs_hist[k] = float(
                diag.get("min_payload_obs_dist", np.nan))
    finally:
        if probe is not None:
            probe_stats = probe.stop()

    return {
        "t": t,
        "p_hist": p_hist,
        "th_hist": th_hist,
        "u_hist": u_hist,
        "wheel_hist": wheel_hist,
        "payload_hist": payload_hist,
        "phase_hist": phase_hist,
        "min_obs_hist": min_obs_hist,
        "min_rr_hist": min_rr_hist,
        "min_payload_obs_hist": min_payload_obs_hist,
        "probe_stats": probe_stats,
        "controller": controller,
    }


def plot_results(sim: dict, args: argparse.Namespace) -> None:
    t = sim["t"]
    p_hist = sim["p_hist"]
    payload_hist = sim["payload_hist"]
    wheel_hist = sim["wheel_hist"]
    phase_hist = sim["phase_hist"]

    min_obs_hist = sim["min_obs_hist"]
    min_rr_hist = sim["min_rr_hist"]
    min_payload_obs_hist = sim["min_payload_obs_hist"]

    ctrl = sim["controller"]
    params = ctrl.params
    obstacle_array = np.asarray(ctrl.obstacle_array)
    payload_goal = np.asarray(ctrl.payload_goal)

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Trajectories
    ax1.plot(p_hist[0, 0, :], p_hist[1, 0, :], "r", label="Robot 1")
    ax1.plot(p_hist[0, 1, :], p_hist[1, 1, :], "b", label="Robot 2")
    ax1.plot(p_hist[0, 2, :], p_hist[1, 2, :], "g", label="Robot 3")
    ax1.plot(payload_hist[0, :], payload_hist[1, :],
             "c--", label="Payload center")

    th = np.linspace(0.0, 2.0 * np.pi, 120)
    for i in range(obstacle_array.shape[1]):
        ax1.plot(
            obstacle_array[0, i] + params.obstacle_radius * np.cos(th),
            obstacle_array[1, i] + params.obstacle_radius * np.sin(th),
            "k",
            linewidth=1.2,
        )

    ax1.plot(payload_hist[0, 0], payload_hist[1, 0],
             "mo", markersize=7, label="Payload start")
    ax1.plot(payload_goal[0], payload_goal[1], "rp",
             markersize=9, label="Payload goal")
    ax1.set_title("Closed-Loop Mock-CV Trajectories")
    ax1.set_xlabel("X [m]")
    ax1.set_ylabel("Y [m]")
    ax1.grid(True)
    ax1.axis("equal")
    ax1.legend(loc="best")

    # Wheel commands
    ax2.plot(t[:-1], wheel_hist[0, :], "r", label="L1")
    ax2.plot(t[:-1], wheel_hist[1, :], "r--", label="R1")
    ax2.plot(t[:-1], wheel_hist[2, :], "b", label="L2")
    ax2.plot(t[:-1], wheel_hist[3, :], "b--", label="R2")
    ax2.plot(t[:-1], wheel_hist[4, :], "g", label="L3")
    ax2.plot(t[:-1], wheel_hist[5, :], "g--", label="R3")
    ax2.set_title("Controller Output: 6 Wheel Velocities")
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Wheel rate [rad/s]")
    ax2.grid(True)
    ax2.legend(loc="best", ncol=2)

    # Safety diagnostics
    ax3.plot(t[:-1], min_obs_hist, "k", label="Min robot-obstacle clearance")
    ax3.plot(t[:-1], min_rr_hist, color=(0.5, 0.2, 0.8),
             label="Min robot-robot clearance")
    ax3.plot(t[:-1], min_payload_obs_hist, color=(0.0, 0.6, 0.6),
             label="Min payload-obstacle clearance")
    ax3.axhline(0.0, color="0.2", linestyle="--",
                linewidth=1.0, label="Safety boundary")
    ax3.set_title("Clearance Diagnostics")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Clearance [m]")
    ax3.grid(True)
    ax3.legend(loc="best")

    # Phase progression
    ax4.step(t[:-1], phase_hist, where="post", color="tab:orange")
    ax4.set_yticks([1, 2])
    ax4.set_yticklabels(["Gather", "Push"])
    ax4.set_title("Controller Phase")
    ax4.set_xlabel("Time [s]")
    ax4.set_ylabel("Phase")
    ax4.grid(True)

    probe_stats = sim.get("probe_stats")
    if probe_stats is not None:
        fig.suptitle(
            "Mock-CV + Plant + TCP Probe | "
            f"lines sent={probe_stats.lines_sent}, "
            f"lines received total={probe_stats.lines_received_total}",
            fontsize=12,
        )

    plt.tight_layout()
    if args.save_plot:
        plt.savefig(args.save_plot, dpi=140)
        print(f"Saved plot to {args.save_plot}")
    if not args.no_plot:
        plt.show()


def print_summary(sim: dict) -> None:
    wheel_hist = sim["wheel_hist"]
    payload_hist = sim["payload_hist"]
    ctrl = sim["controller"]

    final_payload = payload_hist[:, -1]
    goal = np.asarray(ctrl.payload_goal)
    goal_error = float(np.linalg.norm(final_payload - goal))

    print("\nSimulation summary")
    print("------------------")
    print(f"Wheel output shape: {wheel_hist.shape} (expected 6 x T)")
    print(
        f"Final payload center: [{final_payload[0]:.4f}, {final_payload[1]:.4f}] m")
    print(f"Payload goal:         [{goal[0]:.4f}, {goal[1]:.4f}] m")
    print(f"Goal error:           {goal_error:.4f} m")

    min_payload_obs = float(np.nanmin(sim["min_payload_obs_hist"]))
    min_robot_obs = float(np.nanmin(sim["min_obs_hist"]))
    min_rr = float(np.nanmin(sim["min_rr_hist"]))
    print(f"Min payload-obstacle clearance: {min_payload_obs:.4f} m")
    print(f"Min robot-obstacle clearance:   {min_robot_obs:.4f} m")
    print(f"Min robot-robot clearance:      {min_rr:.4f} m")

    probe_stats = sim.get("probe_stats")
    if probe_stats is not None:
        print("\nNetwork probe summary")
        print("---------------------")
        print(f"Broadcast lines sent:      {probe_stats.lines_sent}")
        print(f"Total lines received:      {probe_stats.lines_received_total}")
        for client_idx, count in probe_stats.per_client_received.items():
            print(f"Client {client_idx + 1} lines received: {count}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mock-CV MPC closed-loop simulation")
    parser.add_argument("--tf", type=float, default=50.0,
                        help="Simulation length [s]")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--cv-noise-xy",
        type=float,
        default=0.0,
        help="Mock CV position noise std [m]",
    )
    parser.add_argument(
        "--cv-noise-theta",
        type=float,
        default=0.0,
        help="Mock CV heading noise std [rad]",
    )
    parser.add_argument(
        "--network-probe",
        action="store_true",
        help="Enable loopback TCP probe of command broadcast path",
    )
    parser.add_argument(
        "--probe-host",
        default="127.0.0.1",
        help="Probe BotCommandServer bind host",
    )
    parser.add_argument(
        "--probe-port",
        type=int,
        default=5015,
        help="Probe BotCommandServer TCP port",
    )
    parser.add_argument("--save-plot", default="",
                        help="Optional output image path")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip interactive plotting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sim = run_simulation(args)
    print_summary(sim)
    plot_results(sim, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
