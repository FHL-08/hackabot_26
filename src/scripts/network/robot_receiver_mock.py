import socket
import json

HOST = "127.0.0.1"
PORT = 5000


def fmt_num(value):
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "nan"


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print(f"[RECEIVER] Listening on {HOST}:{PORT}")

    conn, addr = server.accept()
    print(f"[RECEIVER] Connected by {addr}")

    buffer = ""

    try:
        while True:
            data = conn.recv(4096)
            if not data:
                print("[RECEIVER] Client disconnected")
                break

            buffer += data.decode("utf-8")

            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue

                try:
                    packet = json.loads(line)
                    packet_type = packet.get("type")

                    if packet_type == "hello":
                        print("[RECEIVER] Handshake hello received")
                        reply = {
                            "type": "ack",
                            "message": "hello_received"
                        }
                        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))

                    elif packet_type == "marker_data":
                        seq = packet.get("seq")
                        marker_id = packet.get("marker_id")
                        x = packet.get("x")
                        y = packet.get("y")
                        radius = packet.get("radius")
                        markers = packet.get("markers", [])
                        bot_poses = packet.get("bot_poses", [])
                        coord_frame = packet.get("coord_frame")
                        theta_convention = packet.get("theta_convention")
                        units = packet.get("units", "unknown")
                        x_cm = packet.get("x_cm")
                        y_cm = packet.get("y_cm")

                        print(
                            f"[RECEIVER] marker seq={seq} marker={marker_id} "
                            f"x={fmt_num(x)} y={fmt_num(y)} radius={fmt_num(radius)} "
                            f"frame={coord_frame}"
                        )
                        if x_cm is not None and y_cm is not None:
                            print(f"    pose_cm x={fmt_num(x_cm)} y={fmt_num(y_cm)} units={units}")
                        if theta_convention:
                            print(f"    theta_convention={theta_convention}")

                        if bot_poses:
                            print(f"    bot_poses[{len(bot_poses)}]:")
                            for b in bot_poses:
                                print(
                                    "      "
                                    f"id={b.get('marker_id')} "
                                    f"x_cm={fmt_num(b.get('x_cm'))} "
                                    f"y_cm={fmt_num(b.get('y_cm'))} "
                                    f"theta={fmt_num(b.get('theta_rad'))}"
                                )

                        if markers:
                            print(f"    markers[{len(markers)}]:")
                            for m in markers:
                                print(
                                    "      "
                                    f"id={m.get('marker_id')} "
                                    f"x_cm={fmt_num(m.get('x_cm'))} "
                                    f"y_cm={fmt_num(m.get('y_cm'))} "
                                    f"theta={fmt_num(m.get('theta_rad'))}"
                                )

                        obstacles = packet.get("obstacles", [])
                        if obstacles:
                            print(f"    obstacles[{len(obstacles)}]:")
                            for obs in obstacles:
                                print(
                                    "      "
                                    f"id={obs.get('obstacle_id')} "
                                    f"colour={obs.get('colour', 'unknown')} "
                                    f"x_cm={fmt_num(obs.get('x_cm'))} "
                                    f"y_cm={fmt_num(obs.get('y_cm'))} "
                                    f"r_cm={fmt_num(obs.get('radius_cm'))}"
                                )

                        reply = {
                            "type": "ack",
                            "seq": seq,
                            "message": "marker_data_received"
                        }
                        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))

                    elif packet_type == "obstacle_data":
                        seq = packet.get("seq")
                        obstacles = packet.get("obstacles", [])

                        print(f"[RECEIVER] obstacle seq={seq} count={len(obstacles)}")

                        for obs in obstacles:
                            obstacle_id = obs.get("obstacle_id")
                            colour = obs.get("colour", "unknown")
                            x = obs.get("x", 0.0)
                            y = obs.get("y", 0.0)
                            radius = obs.get("radius", 0.0)
                            area = obs.get("area", 0.0)

                            print(
                                f"    obstacle_id={obstacle_id} "
                                f"colour={colour} "
                                f"x={x:.2f} y={y:.2f} "
                                f"radius={radius:.2f} area={area:.2f}"
                            )

                        reply = {
                            "type": "ack",
                            "seq": seq,
                            "message": "obstacle_data_received"
                        }
                        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))

                    else:
                        print(f"[RECEIVER] Unknown packet type: {packet_type}")
                        reply = {
                            "type": "ack",
                            "message": "unknown_packet_type"
                        }
                        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))

                except json.JSONDecodeError:
                    print("[RECEIVER] Bad JSON received")

                except Exception as e:
                    print(f"[RECEIVER] Error while processing packet: {e}")
                    reply = {
                        "type": "ack",
                        "message": f"error: {str(e)}"
                    }
                    try:
                        conn.sendall((json.dumps(reply) + "\n").encode("utf-8"))
                    except Exception:
                        pass

    finally:
        conn.close()
        server.close()
        print("[RECEIVER] Closed")

if __name__ == "__main__":
    main()
