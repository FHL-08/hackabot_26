#!/usr/bin/env python3
"""Drive real robots from simulated state instead of camera input.

This script replaces the tracker/CV input with a local plant simulation while still
broadcasting the real 6-wheel-rate command line to connected robots.

Pipeline:
- simulated robot state -> MPC (TwoPhaseRealtimeController) -> wheel commands
- wheel commands -> TCP broadcast to robot clients on port 5005
- wheel commands -> simulated plant update for next state

Use this when you want to test real robot command streaming without connecting a camera.
"""

from __future__ import annotations

import argparse
import time
from typing import Tuple

import numpy as np

from src.laptop_server import BotCommandServer
from src.mpc import ControllerParams, build_default_controller


def format_broadcast6(values: np.ndarray) -> str:
    """Format [L1,R1,L2,R2,L3,R3] as six-float broadcast line."""
    v = np.asarray(values, dtype=np.float64).reshape(6)
    return (
        f"{v[0]:.6f} {v[1]:.6f} "
        f"{v[2]:.6f} {v[3]:.6f} "
        f"{v[4]:.6f} {v[5]:.6f}"
    )


def default_initial_robot_state() -> Tuple[np.ndarray, np.ndarray]:
    """Return default initial robot positions/headings used by the controller."""
    positions = np.array(
        [
            [-0.01, 0.275, -0.24],
            [-0.68, -0.65, -0.74],
        ],
        dtype=np.float64,
    )
    headings = np.array([0.70, 1.2, 1.5], dtype=np.float64)
    return positions, headings


def plant_step_from_wheel_lr(
    robot_positions: np.ndarray,
    robot_headings: np.ndarray,
    wheel_lr_flat: np.ndarray,
    params: ControllerParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Advance robot states using MATLAB-consistent kinematic mapping."""
    next_positions = robot_positions.copy()
    next_headings = robot_headings.copy()

    wheel = np.asarray(wheel_lr_flat, dtype=np.float64).reshape(6)

    for i in range(3):
        left = float(wheel[2 * i])
        right = float(wheel[2 * i + 1])

        # Internal mapping used by the controller is right-left order.
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

    return next_positions, next_headings


def run(args: argparse.Namespace) -> int:
    """Run closed-loop simulated-state control and broadcast commands to robots."""
    controller = build_default_controller()
    params = controller.params

    # Use the same initial robot state as the MATLAB/Python controller examples.
    robot_positions, robot_headings = default_initial_robot_state()

    bot_server = BotCommandServer(
        host=args.host,
        port=args.port,
        debug=not args.quiet,
        telem_on_change_only=True,
    )
    bot_server.start()

    print("[SIM-DRIVER] Running without camera/tracker input.", flush=True)
    print(
        "[SIM-DRIVER] Controller output is broadcast as six floats: "
        "L1 R1 L2 R2 L3 R3",
        flush=True,
    )
    print(
        f"[SIM-DRIVER] dt={params.dt:.3f}s, challenge limit={params.challenge_time_limit_s:.1f}s, "
        f"requested duration={args.duration_s:.1f}s",
        flush=True,
    )

    t_start = time.monotonic()
    step = 0
    period = float(params.dt)
    next_tick = t_start

    try:
        while True:
            elapsed = time.monotonic() - t_start
            if elapsed >= args.duration_s:
                print("[SIM-DRIVER] Duration reached. Stopping.", flush=True)
                break

            step_out = controller.step(robot_positions, robot_headings)
            cmd = np.asarray(step_out.wheel_rates_lr_flat,
                             dtype=np.float64).reshape(6)

            if float((step_out.diagnostics or {}).get("time_expired", 0.0)) >= 0.5:
                print(
                    "[SIM-DRIVER] Controller mission time expired. "
                    "Sending zero and stopping.",
                    flush=True,
                )
                cmd[:] = 0.0
                line = format_broadcast6(cmd)
                if bot_server.connected_count() > 0:
                    bot_server.broadcast_line(line)
                break

            line = format_broadcast6(cmd)
            if bot_server.connected_count() > 0:
                bot_server.broadcast_line(line)

            robot_positions, robot_headings = plant_step_from_wheel_lr(
                robot_positions,
                robot_headings,
                cmd,
                params,
            )

            if (step % args.log_every_steps) == 0:
                phase = int(step_out.phase)
                min_payload = float(
                    (step_out.diagnostics or {}).get(
                        "min_payload_obs_dist", np.nan)
                )
                clients = bot_server.connected_count()
                cmd_txt = (
                    f"[{cmd[0]:6.2f},{cmd[1]:6.2f},{cmd[2]:6.2f},"
                    f"{cmd[3]:6.2f},{cmd[4]:6.2f},{cmd[5]:6.2f}]"
                )
                print(
                    f"[SIM-DRIVER] t={elapsed:6.2f}s phase={phase} "
                    f"clients={clients} cmd={cmd_txt} "
                    f"min_payload_obs={min_payload:6.3f}",
                    flush=True,
                )

            step += 1

            next_tick += period
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

    except KeyboardInterrupt:
        print("\n[SIM-DRIVER] Ctrl+C received. Stopping.", flush=True)
    finally:
        zero_line = format_broadcast6(np.zeros(6, dtype=np.float64))
        for _ in range(3):
            if bot_server.connected_count() > 0:
                bot_server.broadcast_line(zero_line)
            time.sleep(0.02)
        bot_server.stop()

    return 0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the simulated live robot driver."""
    parser = argparse.ArgumentParser(
        description="Run MPC from simulated state and broadcast real wheel commands to robots"
    )
    parser.add_argument("--host", default="0.0.0.0",
                        help="TCP bind host for robot command server")
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="TCP bind port for robot command server",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=240.0,
        help="How long to run [s] (default matches challenge runtime)",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=15,
        help="Console log cadence in controller steps",
    )
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce debug output from TCP server")
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
