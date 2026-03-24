#!/usr/bin/env python3
"""
Unified tracker -> MPC -> multi-bot TCP orchestrator.

Receives tracker packets on port 5000, computes MPC wheel rates, and broadcasts
newline-delimited bot velocity commands to every connected MONA client on
port 5005 using:
    bot(<id>,<left_rad_s>,<right_rad_s>)
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np

from mpc import (
    build_default_controller,
    cv_state_to_controller_input,
)


MARKER_TO_ROBOT = {
    12: "r1",
    9: "r2",
    4: "r3",
}
ROBOT_ORDER = ("r1", "r2", "r3")
ROBOT_TO_BOT_ID = {
    "r1": 1,
    "r2": 2,
    "r3": 3,
}


def wrap_to_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def controller_runtime_summary(controller: object) -> str:
    params = getattr(controller, "params", None)
    obstacle_array = getattr(controller, "obstacle_array", None)
    payload_goal = getattr(controller, "payload_goal", None)
    payload_center = getattr(controller, "payload_center", None)

    if params is None or obstacle_array is None or payload_goal is None or payload_center is None:
        return "[ORCH] MPC summary unavailable"

    try:
        obs_txt = np.array2string(np.asarray(
            obstacle_array), precision=3, suppress_small=True)
        return (
            "[ORCH] MPC loaded: "
            f"obstacle_radius={float(params.obstacle_radius):.4f}, "
            f"d_safe_payload_obs={float(getattr(controller, 'd_safe_payload_obs', 0.0)):.4f}, "
            f"payload_start={np.asarray(payload_center).tolist()}, "
            f"payload_goal={np.asarray(payload_goal).tolist()}, "
            f"obstacles={obs_txt}"
        )
    except Exception:
        return "[ORCH] MPC loaded (summary format failed)"


@dataclass
class ClientConn:
    conn: socket.socket
    addr: tuple[str, int]


class BotCommandHub:
    def __init__(self, host: str, port: int, *, quiet: bool = False) -> None:
        self.host = host
        self.port = port
        self.quiet = quiet

        self._server: Optional[socket.socket] = None
        self._running = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None
        self._clients: Dict[int, ClientConn] = {}
        self._next_client_id = 1
        self._lock = threading.Lock()

    def start(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(8)
        server.settimeout(1.0)

        self._server = server
        self._running.set()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True)
        self._accept_thread.start()
        self._log(f"[BOT-HUB] Listening on {self.host}:{self.port}")

    def stop(self) -> None:
        self._running.clear()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
            self._server = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None

        with self._lock:
            items = list(self._clients.items())
            self._clients.clear()
        for _, client in items:
            try:
                client.conn.close()
            except OSError:
                pass

    def connected_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def broadcast_wheel_vector(self, lr_flat: Iterable[float]) -> list[str]:
        values = np.asarray(list(lr_flat), dtype=np.float64).reshape(6)
        lines = [
            f"bot(1,{values[0]:.6f},{values[1]:.6f})",
            f"bot(2,{values[2]:.6f},{values[3]:.6f})",
            f"bot(3,{values[4]:.6f},{values[5]:.6f})",
        ]
        self.broadcast_lines(lines)
        return lines

    def broadcast_zero(self) -> list[str]:
        return self.broadcast_wheel_vector(np.zeros(6, dtype=np.float64))

    def broadcast_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        payload = "".join(
            f"{line.rstrip()}\n" for line in lines).encode("utf-8")

        with self._lock:
            snapshot = list(self._clients.items())

        dead: list[int] = []
        for client_id, client in snapshot:
            try:
                client.conn.sendall(payload)
            except OSError:
                dead.append(client_id)

        if dead:
            with self._lock:
                for client_id in dead:
                    client = self._clients.pop(client_id, None)
                    if client is None:
                        continue
                    try:
                        client.conn.close()
                    except OSError:
                        pass
                    self._log(f"[BOT-HUB] Dropped client #{client_id}")

    def _accept_loop(self) -> None:
        assert self._server is not None
        while self._running.is_set():
            try:
                conn, addr = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            conn.settimeout(1.0)
            with self._lock:
                client_id = self._next_client_id
                self._next_client_id += 1
                self._clients[client_id] = ClientConn(conn=conn, addr=addr)
            self._log(
                f"[BOT-HUB] Client #{client_id} connected: {addr[0]}:{addr[1]}")

            t = threading.Thread(
                target=self._client_recv_loop,
                args=(client_id, conn, addr),
                daemon=True,
            )
            t.start()

    def _client_recv_loop(self, client_id: int, conn: socket.socket, addr: tuple[str, int]) -> None:
        peer = f"{addr[0]}:{addr[1]}"
        buffer = b""
        try:
            while self._running.is_set():
                try:
                    chunk = conn.recv(1024)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").strip()
                    if text:
                        self._log(f"[BOT {peer}] {text}")
        except OSError:
            pass
        finally:
            with self._lock:
                client = self._clients.pop(client_id, None)
            if client is not None:
                try:
                    client.conn.close()
                except OSError:
                    pass
            self._log(f"[BOT-HUB] Client #{client_id} disconnected: {peer}")

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, flush=True)


class SwarmOrchestrator:
    def __init__(
        self,
        *,
        tracker_host: str,
        tracker_port: int,
        bot_host: str,
        bot_port: int,
        theta_offset_rad: float,
        stale_timeout_s: float,
        zero_send_hz: float,
        log_file: Optional[Path],
        quiet: bool,
    ) -> None:
        self.tracker_host = tracker_host
        self.tracker_port = tracker_port
        self.theta_offset_rad = float(theta_offset_rad)
        self.stale_timeout_s = float(stale_timeout_s)
        self.zero_send_period_s = 1.0 / max(float(zero_send_hz), 1.0)
        self.log_file = log_file
        self.quiet = quiet

        self.controller = build_default_controller()
        self._log(controller_runtime_summary(self.controller))
        self.bot_hub = BotCommandHub(bot_host, bot_port, quiet=quiet)

        self._running = threading.Event()
        self._last_tracker_monotonic = 0.0
        self._last_seq: Optional[int] = None
        self._watchdog_thread: Optional[threading.Thread] = None

        self._log_fp = None
        if self.log_file is not None:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_fp = self.log_file.open("a", encoding="utf-8")

    def run(self) -> None:
        self.bot_hub.start()
        self._running.set()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.tracker_host, self.tracker_port))
        server.listen(1)
        server.settimeout(1.0)

        self._log(
            f"[TRACKER-HUB] Listening on {self.tracker_host}:{self.tracker_port}")
        self._log("[TRACKER-HUB] Waiting for tracker_sender connection...")

        try:
            while self._running.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                peer = f"{addr[0]}:{addr[1]}"
                self._log(f"[TRACKER-HUB] Connected: {peer}")
                try:
                    self._handle_tracker_conn(conn)
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
                    self._log(f"[TRACKER-HUB] Disconnected: {peer}")
                    self._last_tracker_monotonic = 0.0
                    self._last_seq = None
        finally:
            self._running.clear()
            try:
                server.close()
            except OSError:
                pass
            if self._watchdog_thread is not None:
                self._watchdog_thread.join(timeout=1.0)
            self.bot_hub.stop()
            if self._log_fp is not None:
                self._log_fp.close()

    def _watchdog_loop(self) -> None:
        while self._running.is_set():
            now = time.monotonic()
            age = now - \
                self._last_tracker_monotonic if self._last_tracker_monotonic > 0.0 else float(
                    "inf")
            if age > self.stale_timeout_s:
                self.bot_hub.broadcast_zero()
            time.sleep(self.zero_send_period_s)

    def _handle_tracker_conn(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buffer = ""
        while self._running.is_set():
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            if not data:
                break
            buffer += data.decode("utf-8", errors="replace")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    packet = json.loads(line)
                except json.JSONDecodeError:
                    self._send_json(
                        conn, {"type": "ack", "message": "bad_json"})
                    continue

                ptype = packet.get("type")
                if ptype == "hello":
                    self._send_json(
                        conn, {"type": "ack", "message": "hello_received"})
                    continue

                if ptype != "marker_data":
                    self._send_json(
                        conn, {"type": "ack", "message": "unknown_packet_type"})
                    continue

                seq = int(packet.get("seq", -1))
                if self._last_seq is not None and seq == self._last_seq:
                    self._send_json(
                        conn, {"type": "ack", "seq": seq, "message": "duplicate_seq_ignored"})
                    continue
                self._last_seq = seq

                tick = self._process_marker_packet(packet)
                self._send_json(
                    conn, {"type": "ack", "seq": seq, "message": "marker_data_received"})
                self._log_tick(tick)
                self._log_jsonl(tick)

    def _process_marker_packet(self, packet: dict) -> dict:
        now = time.monotonic()
        self._last_tracker_monotonic = now

        seq = int(packet.get("seq", -1))
        timestamp = float(packet.get("timestamp", time.time()))
        bot_poses = packet.get("bot_poses") or packet.get("markers") or []

        state_by_robot: Dict[str, Dict[str, float]] = {}
        for pose in bot_poses:
            try:
                marker_id = int(pose["marker_id"])
            except Exception:
                continue
            robot_name = MARKER_TO_ROBOT.get(marker_id)
            if robot_name is None:
                continue
            try:
                x_m = float(pose["x_cm"]) / 100.0
                y_m = float(pose["y_cm"]) / 100.0
                theta = wrap_to_pi(
                    float(pose["theta_rad"]) + self.theta_offset_rad)
            except Exception:
                continue
            state_by_robot[robot_name] = {
                "x": x_m,
                "y": y_m,
                "theta": theta,
            }

        missing = [r for r in ROBOT_ORDER if r not in state_by_robot]
        if missing:
            cmd_vec = np.zeros(6, dtype=np.float64)
            phase = int(getattr(self.controller, "phase", 0))
            diagnostics = {
                "reason_missing_markers": ",".join(missing),
                "time_expired": 0.0,
            }
            lines = self.bot_hub.broadcast_wheel_vector(cmd_vec)
            return {
                "seq": seq,
                "timestamp": timestamp,
                "status": "failsafe_missing_markers",
                "missing": missing,
                "phase": phase,
                "commands_lr_flat": cmd_vec.tolist(),
                "command_lines": lines,
                "diagnostics": diagnostics,
                "connected_bots": self.bot_hub.connected_count(),
            }

        robot_pos, robot_theta = cv_state_to_controller_input(
            state_by_robot,
            robot_order=ROBOT_ORDER,
        )
        step_output = self.controller.step(robot_pos, robot_theta)
        cmd_vec = np.asarray(step_output.wheel_rates_lr_flat,
                             dtype=np.float64).reshape(6)
        diagnostics = dict(step_output.diagnostics or {})

        time_expired = float(diagnostics.get("time_expired", 0.0))
        status = "ok"
        if time_expired >= 0.5:
            status = "failsafe_time_expired"
            cmd_vec = np.zeros(6, dtype=np.float64)

        lines = self.bot_hub.broadcast_wheel_vector(cmd_vec)

        return {
            "seq": seq,
            "timestamp": timestamp,
            "status": status,
            "missing": [],
            "phase": int(step_output.phase),
            "states_m": state_by_robot,
            "commands_lr_flat": cmd_vec.tolist(),
            "command_lines": lines,
            "diagnostics": diagnostics,
            "connected_bots": self.bot_hub.connected_count(),
        }

    def _send_json(self, conn: socket.socket, payload: dict) -> None:
        msg = json.dumps(payload, separators=(",", ":")) + "\n"
        conn.sendall(msg.encode("utf-8"))

    def _log_tick(self, tick: dict) -> None:
        if self.quiet:
            return
        cmd = tick.get("commands_lr_flat", [0, 0, 0, 0, 0, 0])
        print(
            "[ORCH] "
            f"seq={tick.get('seq')} "
            f"status={tick.get('status')} "
            f"phase={tick.get('phase')} "
            f"bots={tick.get('connected_bots')} "
            f"L1={cmd[0]:.3f} R1={cmd[1]:.3f} "
            f"L2={cmd[2]:.3f} R2={cmd[3]:.3f} "
            f"L3={cmd[4]:.3f} R3={cmd[5]:.3f} "
            f"min_payload_obs={float((tick.get('diagnostics') or {}).get('min_payload_obs_dist', float('nan'))):.4f}",
            flush=True,
        )

    def _log_jsonl(self, tick: dict) -> None:
        if self._log_fp is None:
            return
        self._log_fp.write(json.dumps(tick, separators=(",", ":")) + "\n")
        self._log_fp.flush()

    def _log(self, msg: str) -> None:
        if not self.quiet:
            print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified tracker->MPC->multi-bot orchestrator",
    )
    parser.add_argument("--tracker-host", default="0.0.0.0",
                        help="Tracker server bind host")
    parser.add_argument("--tracker-port", type=int,
                        default=5000, help="Tracker server bind port")
    parser.add_argument("--bot-host", default="0.0.0.0",
                        help="Bot command server bind host")
    parser.add_argument("--bot-port", type=int, default=5005,
                        help="Bot command server bind port")
    parser.add_argument(
        "--theta-offset-rad",
        type=float,
        default=0.0,
        help="Global heading offset applied to tracker theta before MPC",
    )
    parser.add_argument(
        "--stale-timeout-s",
        type=float,
        default=0.20,
        help="If no fresh tracker packet for this long, broadcast zero commands",
    )
    parser.add_argument(
        "--zero-send-hz",
        type=float,
        default=30.0,
        help="Zero-command broadcast frequency during stale/disconnected tracker periods",
    )
    parser.add_argument(
        "--log-file",
        default="src/generated/logs/swarm_orchestrator_log.jsonl",
        help="JSONL output path (empty string disables file logging)",
    )
    parser.add_argument("--quiet", action="store_true",
                        help="Reduce console logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file) if args.log_file else None
    orch = SwarmOrchestrator(
        tracker_host=args.tracker_host,
        tracker_port=args.tracker_port,
        bot_host=args.bot_host,
        bot_port=args.bot_port,
        theta_offset_rad=args.theta_offset_rad,
        stale_timeout_s=args.stale_timeout_s,
        zero_send_hz=args.zero_send_hz,
        log_file=log_path,
        quiet=args.quiet,
    )
    try:
        orch.run()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
