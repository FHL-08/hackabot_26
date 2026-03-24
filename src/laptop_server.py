#!/usr/bin/env python3
"""
communication_tcp - laptop TCP server for MONA velocity control.

Manual mode (default):
- Broadcast six-float lines to all connected bots:
    <b1_L> <b1_R> <b2_L> <b2_R> <b3_L> <b3_R>
- Supports legacy bot(id,left,right) lines for compatibility.

Auto mode (--auto-from-tracker):
- Receives tracker packets on port 5000.
- Runs MPC each tracker frame.
- Broadcasts six-float command lines to all connected bots on port 5005.

Important:
- Run this while the laptop is connected to same local network as robots Wi-Fi (warn-only check).
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from mpc import build_default_controller, cv_state_to_controller_input


DEFAULT_BOT_PORT = 5005
DEFAULT_TRACKER_PORT = 5000
YOUR_WIFI_SSID = "TP-Link_6C24"

MARKER_TO_ROBOT = {
    12: "r1",
    9: "r2",
    4: "r3",
}
ROBOT_ORDER = ("r1", "r2", "r3")

_STATE_RE = re.compile(
    r"^(?P<bot>\w+)\s+state:\s*omegaL=(?P<wl>[\d.eE+-]+)\s*omegaR=(?P<wr>[\d.eE+-]+)\s*"
    r"pwmL=(?P<pl>-?\d+)\s*pwmR=(?P<pr>-?\d+)\s*$",
)

_BROADCAST6_RE = re.compile(
    r"^[\s]*"
    r"(?P<a>[\d.eE+-]+)\s+"
    r"(?P<b>[\d.eE+-]+)\s+"
    r"(?P<c>[\d.eE+-]+)\s+"
    r"(?P<d>[\d.eE+-]+)\s+"
    r"(?P<e>[\d.eE+-]+)\s+"
    r"(?P<f>[\d.eE+-]+)\s*$"
)


def _dbg(msg: str, *, debug: bool) -> None:
    if debug:
        ts = time.strftime("%H:%M:%S")
        print(f"[dbg {ts}] {msg}", flush=True)


def _run_command(cmd: List[str]) -> Tuple[int, str]:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, shell=False)
        return result.returncode, result.stdout or ""
    except OSError:
        return 1, ""


def _current_wifi_ssid() -> Optional[str]:
    code, stdout = _run_command(["netsh", "wlan", "show", "interfaces"])
    if code != 0:
        return None
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("SSID") and "BSSID" not in stripped:
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                return parts[1].strip()
    return None


def _laptop_lan_ipv4() -> Optional[str]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        sock.connect(("192.0.2.1", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return None


def _print_wifi_hint(*, debug: bool, expected_ssid: str) -> None:
    ssid = _current_wifi_ssid()
    if ssid:
        print(f"[wifi-check] Current Wi-Fi SSID: {ssid}", flush=True)
        if ssid != expected_ssid:
            print(
                f"[wifi-check] WARNING: expected SSID '{expected_ssid}', got '{ssid}'. "
                "Connect to '{expected_ssid}' before running bots.",
                flush=True,
            )
    else:
        print("[wifi-check] Could not read Wi-Fi SSID via netsh.", flush=True)

    ip = _laptop_lan_ipv4()
    if ip is None:
        print(
            "[wifi-check] Could not guess LAN IPv4 - MONA may not reach this laptop.",
            flush=True,
        )
        return
    if ip.startswith("127."):
        print(
            f"[wifi-check] Outbound IP is loopback ({ip}) - connect Wi-Fi/Ethernet.",
            flush=True,
        )
        return
    print(
        f"[wifi-check] Laptop LAN IPv4: {ip} (MONA LAPTOP_HOST should match)",
        flush=True,
    )
    _dbg(f"wifi-check OK ({ip})", debug=debug)


def _format_bot(bot_id: int, left: float, right: float) -> str:
    return f"bot({bot_id},{left},{right})"


def _format_broadcast6(
    b1_l: float,
    b1_r: float,
    b2_l: float,
    b2_r: float,
    b3_l: float,
    b3_r: float,
) -> str:
    return (
        f"{float(b1_l):.6f} {float(b1_r):.6f} "
        f"{float(b2_l):.6f} {float(b2_r):.6f} "
        f"{float(b3_l):.6f} {float(b3_r):.6f}"
    )


def _shortcut_broadcast6(
    line: str,
    *,
    fwd_l: float,
    fwd_r: float,
    stop_l: float,
    stop_r: float,
    turn_a_l: float,
    turn_a_r: float,
    turn_d_l: float,
    turn_d_r: float,
) -> Optional[str]:
    s = line.strip()
    if len(s) != 1:
        return None
    c = s[0].upper()
    if c == "W":
        return _format_broadcast6(fwd_l, fwd_r, fwd_l, fwd_r, fwd_l, fwd_r)
    if c == "S":
        return _format_broadcast6(stop_l, stop_r, stop_l, stop_r, stop_l, stop_r)
    if c == "A":
        return _format_broadcast6(turn_a_l, turn_a_r, turn_a_l, turn_a_r, turn_a_l, turn_a_r)
    if c == "D":
        return _format_broadcast6(turn_d_l, turn_d_r, turn_d_l, turn_d_r, turn_d_l, turn_d_r)
    return None


def _shortcut_to_bot_legacy(
    line: str,
    *,
    bot_id: int,
    fwd_l: float,
    fwd_r: float,
    stop_l: float,
    stop_r: float,
    turn_a_l: float,
    turn_a_r: float,
    turn_d_l: float,
    turn_d_r: float,
) -> Optional[str]:
    s = line.strip()
    if len(s) != 1:
        return None
    c = s[0].upper()
    if c == "W":
        return _format_bot(bot_id, fwd_l, fwd_r)
    if c == "S":
        return _format_bot(bot_id, stop_l, stop_r)
    if c == "A":
        return _format_bot(bot_id, turn_a_l, turn_a_r)
    if c == "D":
        return _format_bot(bot_id, turn_d_l, turn_d_r)
    return None


def _wrap_to_pi(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def _parse_tracker_pose_map(packet: dict, theta_offset_rad: float) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
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
            theta = _wrap_to_pi(float(pose["theta_rad"]) + theta_offset_rad)
        except Exception:
            continue
        state_by_robot[robot_name] = {"x": x_m, "y": y_m, "theta": theta}

    missing = [r for r in ROBOT_ORDER if r not in state_by_robot]
    return state_by_robot, missing


def _controller_runtime_summary(controller: object) -> str:
    params = getattr(controller, "params", None)
    obstacle_array = getattr(controller, "obstacle_array", None)
    payload_goal = getattr(controller, "payload_goal", None)
    payload_center = getattr(controller, "payload_center", None)

    if params is None or obstacle_array is None or payload_goal is None or payload_center is None:
        return "[AUTO] MPC summary unavailable"

    try:
        obs_txt = np.array2string(np.asarray(
            obstacle_array), precision=3, suppress_small=True)
        return (
            "[AUTO] MPC loaded: "
            f"obstacle_radius={float(params.obstacle_radius):.4f}, "
            f"d_safe_payload_obs={float(getattr(controller, 'd_safe_payload_obs', 0.0)):.4f}, "
            f"payload_start={np.asarray(payload_center).tolist()}, "
            f"payload_goal={np.asarray(payload_goal).tolist()}, "
            f"obstacles={obs_txt}"
        )
    except Exception:
        return "[AUTO] MPC loaded (summary format failed)"


class BotCommandServer:
    def __init__(self, host: str, port: int, *, debug: bool, telem_on_change_only: bool) -> None:
        self.host = host
        self.port = port
        self.debug = debug
        self.telem_on_change_only = telem_on_change_only

        self.server: Optional[socket.socket] = None
        self.clients_lock = threading.Lock()
        self.clients: List[socket.socket] = []
        self._running = threading.Event()
        self._accept_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((self.host, self.port))
        self.server.listen(8)
        self.server.settimeout(1.0)

        self._running.set()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, daemon=True)
        self._accept_thread.start()

        _dbg(
            f"bound socket fd={self.server.fileno()} {self.host}:{self.port}", debug=self.debug)
        print(
            f"communication_tcp: listening on {self.host}:{self.port}", flush=True)
        print("Waiting for MONA client(s) to connect...", flush=True)

    def stop(self) -> None:
        self._running.clear()
        if self.server is not None:
            try:
                self.server.close()
            except OSError:
                pass
            self.server = None
        if self._accept_thread is not None:
            self._accept_thread.join(timeout=1.0)
            self._accept_thread = None

        with self.clients_lock:
            for c in self.clients:
                try:
                    c.close()
                except OSError:
                    pass
            self.clients.clear()

    def connected_count(self) -> int:
        with self.clients_lock:
            return len(self.clients)

    def broadcast_line(self, line: str) -> None:
        payload = (line.rstrip("\r\n") + "\n").encode("utf-8")
        with self.clients_lock:
            conns = list(self.clients)
        dead: List[socket.socket] = []
        for c in conns:
            try:
                c.sendall(payload)
            except OSError:
                dead.append(c)
        for c in dead:
            self._remove_client(c)

    def _remove_client(self, sock: socket.socket) -> None:
        with self.clients_lock:
            if sock in self.clients:
                self.clients.remove(sock)
        try:
            sock.close()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        assert self.server is not None
        while self._running.is_set():
            try:
                conn, addr = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            peer = f"{addr[0]}:{addr[1]}"
            _dbg(
                f"accept() -> peer={peer} fd={conn.fileno()}", debug=self.debug)
            with self.clients_lock:
                self.clients.append(conn)
                n = len(self.clients)
            print(f"Connected {peer} ({n} client(s))", flush=True)
            threading.Thread(
                target=self._recv_lines,
                args=(conn, peer),
                daemon=True,
            ).start()

    def _recv_lines(self, sock: socket.socket, peer: str) -> None:
        buf = b""
        last_state: Dict[str, Tuple[str, str, str, str]] = {}
        try:
            while self._running.is_set():
                chunk = sock.recv(1024)
                if not chunk:
                    _dbg(f"{peer} recv: empty chunk (peer closed TCP)",
                         debug=self.debug)
                    print(f"\n[{peer}] disconnected", flush=True)
                    return
                _dbg(f"{peer} recv: got {len(chunk)} raw bytes", debug=self.debug)
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    text = line.decode("utf-8", errors="replace").rstrip("\r")
                    _dbg(
                        f"{peer} parsed line ({len(text)} chars): {text!r}", debug=self.debug)
                    if self.telem_on_change_only:
                        m = _STATE_RE.match(text)
                        if m:
                            bot = m.group("bot")
                            key = (m.group("wl"), m.group("wr"),
                                   m.group("pl"), m.group("pr"))
                            if last_state.get(bot) != key:
                                last_state[bot] = key
                                print(f"[{peer}] {text}", flush=True)
                            continue
                    print(f"[{peer}] {text}", flush=True)
        except OSError as e:
            print(f"\n[{peer}] recv error: {e}", flush=True)
        finally:
            self._remove_client(sock)
            print(f"[{peer}] removed from broadcast list", flush=True)


class TrackerAutoController:
    def __init__(
        self,
        *,
        bot_server: BotCommandServer,
        host: str,
        port: int,
        control_hz: float,
        stale_timeout_s: float,
        theta_offset_rad: float,
        debug: bool,
    ) -> None:
        self.bot_server = bot_server
        self.host = host
        self.port = port
        self.control_hz = max(float(control_hz), 1.0)
        self.stale_timeout_s = float(stale_timeout_s)
        self.theta_offset_rad = float(theta_offset_rad)
        self.debug = debug

        self.controller = build_default_controller()
        print(_controller_runtime_summary(self.controller), flush=True)
        self._running = threading.Event()
        self._server_thread: Optional[threading.Thread] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._last_tracker_packet_mono = 0.0
        self._last_seq: Optional[int] = None

    def start(self) -> None:
        self._running.set()
        self._server_thread = threading.Thread(
            target=self._server_loop, daemon=True)
        self._server_thread.start()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()
        print(
            f"[AUTO] Tracker listener on {self.host}:{self.port}", flush=True)
        print("[AUTO] Running frame-synced MPC control from tracker data.", flush=True)

    def stop(self) -> None:
        self._running.clear()
        if self._server_thread is not None:
            self._server_thread.join(timeout=1.0)
            self._server_thread = None
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=1.0)
            self._watchdog_thread = None

    def _server_loop(self) -> None:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(1)
        server.settimeout(1.0)
        print("[AUTO] Waiting for tracker_sender connection...", flush=True)
        try:
            while self._running.is_set():
                try:
                    conn, addr = server.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                peer = f"{addr[0]}:{addr[1]}"
                print(f"[AUTO] tracker connected: {peer}", flush=True)
                try:
                    self._handle_tracker_conn(conn)
                finally:
                    try:
                        conn.close()
                    except OSError:
                        pass
                    print(f"[AUTO] tracker disconnected: {peer}", flush=True)
                    self._last_tracker_packet_mono = 0.0
                    self._last_seq = None
        finally:
            try:
                server.close()
            except OSError:
                pass

    def _watchdog_loop(self) -> None:
        period = 1.0 / self.control_hz
        zero_line = _format_broadcast6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        while self._running.is_set():
            age = (
                time.monotonic() - self._last_tracker_packet_mono
                if self._last_tracker_packet_mono > 0.0
                else float("inf")
            )
            if age > self.stale_timeout_s:
                if self.bot_server.connected_count() > 0:
                    self.bot_server.broadcast_line(zero_line)
            time.sleep(period)

    def _send_json(self, conn: socket.socket, payload: dict) -> None:
        msg = json.dumps(payload, separators=(",", ":")) + "\n"
        conn.sendall(msg.encode("utf-8"))

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

                cmd_line, status, phase = self._run_mpc_step(packet)
                if self.bot_server.connected_count() > 0:
                    self.bot_server.broadcast_line(cmd_line)
                self._send_json(
                    conn, {"type": "ack", "seq": seq, "message": "marker_data_received"})
                if not self.debug:
                    print(
                        f"[AUTO] seq={seq} status={status} phase={phase} cmd={cmd_line}", flush=True)
                else:
                    _dbg(
                        f"[AUTO] seq={seq} status={status} phase={phase} cmd={cmd_line}", debug=True)

    def _run_mpc_step(self, packet: dict) -> Tuple[str, str, int]:
        self._last_tracker_packet_mono = time.monotonic()
        state_by_robot, missing = _parse_tracker_pose_map(
            packet, self.theta_offset_rad)
        if missing:
            return _format_broadcast6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), f"missing_{','.join(missing)}", int(
                getattr(self.controller, "phase", 0)
            )

        robot_pos, robot_theta = cv_state_to_controller_input(
            state_by_robot, robot_order=ROBOT_ORDER)
        step_out = self.controller.step(robot_pos, robot_theta)
        cmd = np.asarray(step_out.wheel_rates_lr_flat,
                         dtype=np.float64).reshape(6)

        diagnostics = step_out.diagnostics or {}
        if float(diagnostics.get("time_expired", 0.0)) >= 0.5:
            return _format_broadcast6(0.0, 0.0, 0.0, 0.0, 0.0, 0.0), "time_expired", int(step_out.phase)

        min_payload_obs_dist = diagnostics.get("min_payload_obs_dist")
        if self.debug and min_payload_obs_dist is not None:
            _dbg(
                f"[AUTO] payload clearance={float(min_payload_obs_dist):.4f} m", debug=True)

        return _format_broadcast6(cmd[0], cmd[1], cmd[2], cmd[3], cmd[4], cmd[5]), "ok", int(step_out.phase)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="communication_tcp laptop TCP server (multi-bot)")
    p.add_argument("--host", default="0.0.0.0",
                   help="Bot command bind address")
    p.add_argument("--port", type=int, default=DEFAULT_BOT_PORT,
                   help="Bot command TCP port")
    p.add_argument("--quiet", action="store_true", help="Turn off debug lines")
    p.add_argument("--no-wifi-check", action="store_true",
                   help="Skip Wi-Fi check")
    p.add_argument("--expected-ssid", default=DEFAULT_EXPECTED_SSID,
                   help="Expected control SSID (warn-only)")
    p.add_argument(
        "--stream-telem",
        action="store_true",
        help="Print every MONA state line (default TTY: hide duplicate state lines)",
    )
    p.add_argument("--bot-id", type=int, default=1,
                   help="Legacy bot(id,...) shortcut target")
    p.add_argument(
        "--legacy-shortcuts",
        action="store_true",
        help="W/S/A/D send bot(bot-id,...) instead of six-float broadcast",
    )

    p.add_argument("--fwd-l", type=float, default=20.0,
                   help="W key: left wheel rad/s")
    p.add_argument("--fwd-r", type=float, default=20.0,
                   help="W key: right wheel rad/s")
    p.add_argument("--stop-l", type=float, default=0.0,
                   help="S key: left target")
    p.add_argument("--stop-r", type=float, default=0.0,
                   help="S key: right target")
    p.add_argument("--turn-a-l", type=float,
                   default=15.0, help="A key: left rad/s")
    p.add_argument("--turn-a-r", type=float, default=20.0,
                   help="A key: right rad/s")
    p.add_argument("--turn-d-l", type=float,
                   default=20.0, help="D key: left rad/s")
    p.add_argument("--turn-d-r", type=float, default=15.0,
                   help="D key: right rad/s")

    p.add_argument("--auto-from-tracker", action="store_true",
                   help="Enable tracker->MPC automatic control mode")
    p.add_argument("--tracker-host", default="0.0.0.0",
                   help="Tracker listener bind address")
    p.add_argument("--tracker-port", type=int,
                   default=DEFAULT_TRACKER_PORT, help="Tracker listener TCP port")
    p.add_argument("--theta-offset-rad", type=float, default=0.0,
                   help="Global heading offset before MPC")
    p.add_argument("--stale-timeout-s", type=float, default=0.20,
                   help="Tracker stale timeout for zero fail-safe")
    p.add_argument("--control-hz", type=float, default=30.0,
                   help="Fail-safe zero broadcast rate when tracker stale")
    return p.parse_args()


def _run_manual_loop(args: argparse.Namespace, bot_server: BotCommandServer, *, debug: bool) -> None:
    telem_on_change_only = sys.stdin.isatty() and not args.stream_telem
    if telem_on_change_only:
        print(
            "[info] Duplicate MONA `state:` lines are hidden (use --stream-telem for all).",
            flush=True,
        )

    if args.legacy_shortcuts:
        print(
            f"[info] Shortcuts: W/S/A/D -> {_format_bot(args.bot_id, args.fwd_l, args.fwd_r)} style "
            f"(or paste bot(...) / six floats)",
            flush=True,
        )
    else:
        ex = _format_broadcast6(args.fwd_l, args.fwd_r,
                                args.fwd_l, args.fwd_r, args.fwd_l, args.fwd_r)
        print(
            f"[info] Shortcuts: W/S/A/D -> six floats (all bots same preset). Example W: {ex}",
            flush=True,
        )
        print("[info] Or paste six floats: b1_L b1_R b2_L b2_R b3_L b3_R", flush=True)

    while True:
        try:
            line = input("send> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip():
            break

        raw = line.strip()
        out: Optional[str] = None

        if raw.lower().startswith("bot("):
            out = raw
        elif _BROADCAST6_RE.match(raw):
            out = raw
        elif args.legacy_shortcuts:
            out = _shortcut_to_bot_legacy(
                line,
                bot_id=args.bot_id,
                fwd_l=args.fwd_l,
                fwd_r=args.fwd_r,
                stop_l=args.stop_l,
                stop_r=args.stop_r,
                turn_a_l=args.turn_a_l,
                turn_a_r=args.turn_a_r,
                turn_d_l=args.turn_d_l,
                turn_d_r=args.turn_d_r,
            )
        else:
            out = _shortcut_broadcast6(
                line,
                fwd_l=args.fwd_l,
                fwd_r=args.fwd_r,
                stop_l=args.stop_l,
                stop_r=args.stop_r,
                turn_a_l=args.turn_a_l,
                turn_a_r=args.turn_a_r,
                turn_d_l=args.turn_d_l,
                turn_d_r=args.turn_d_r,
            )

        if out is None:
            print("Unknown: use W/S/A/D or six floats or bot(id,l,r).", flush=True)
            continue

        n = bot_server.connected_count()
        if n == 0:
            print("[warn] No TCP clients connected - nothing sent.", flush=True)
            continue

        _dbg(f"broadcast to {n} client(s): {out!r}", debug=debug)
        bot_server.broadcast_line(out)
        _dbg("sendall() OK", debug=debug)


def main() -> int:
    args = parse_args()
    debug = not args.quiet
    telem_on_change_only = sys.stdin.isatty() and not args.stream_telem

    if not args.no_wifi_check:
        _print_wifi_hint(debug=debug, expected_ssid=args.expected_ssid)

    bot_server = BotCommandServer(
        args.host,
        args.port,
        debug=debug,
        telem_on_change_only=telem_on_change_only,
    )
    bot_server.start()

    auto: Optional[TrackerAutoController] = None
    if args.auto_from_tracker:
        auto = TrackerAutoController(
            bot_server=bot_server,
            host=args.tracker_host,
            port=args.tracker_port,
            control_hz=args.control_hz,
            stale_timeout_s=args.stale_timeout_s,
            theta_offset_rad=args.theta_offset_rad,
            debug=debug,
        )
        auto.start()

    try:
        if args.auto_from_tracker:
            print("[AUTO] Press Ctrl+C to stop.", flush=True)
            while True:
                time.sleep(3600)
        else:
            if sys.stdin.isatty():
                _run_manual_loop(args, bot_server, debug=debug)
            else:
                print(
                    "[info] stdin is not a terminal - keeping TCP open; run in terminal for interactive commands.",
                    flush=True,
                )
                while True:
                    time.sleep(3600)
    except KeyboardInterrupt:
        print(flush=True)
    finally:
        _dbg("closing auto/tracker + clients + server", debug=debug)
        if auto is not None:
            auto.stop()
        bot_server.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
