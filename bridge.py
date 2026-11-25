"""Minimal localhost bridge between Isaac Lab (Python 3.12) and GR00T N1 (its own Python 3.10 venv).

Wire format per message: 8-byte header (json_len, npz_len, network order) + JSON + .npz bytes.
Arrays are loaded with allow_pickle=False, so no code can travel over the socket.
Only stdlib + numpy, so it imports cleanly in both environments.
"""

import io
import json
import socket
import struct

import numpy as np


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("bridge peer closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def send_msg(sock: socket.socket, header: dict, arrays: dict | None = None) -> None:
    payload = b""
    if arrays:
        bio = io.BytesIO()
        np.savez(bio, **arrays)
        payload = bio.getvalue()
    head = json.dumps(header).encode()
    sock.sendall(struct.pack("!II", len(head), len(payload)) + head + payload)


def recv_msg(sock: socket.socket) -> tuple[dict, dict]:
    head_len, payload_len = struct.unpack("!II", _recv_exact(sock, 8))
    header = json.loads(_recv_exact(sock, head_len))
    arrays = {}
    if payload_len:
        with np.load(io.BytesIO(_recv_exact(sock, payload_len)), allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
    return header, arrays


class Gr00tClient:
    """Isaac-side client: send an observation, get a 16-step action chunk back."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5566, timeout_s: float = 120.0):
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def ping(self) -> dict:
        send_msg(self.sock, {"cmd": "ping"})
        return recv_msg(self.sock)[0]

    def get_action(self, instruction: str, obs: dict) -> tuple[dict, dict]:
        send_msg(self.sock, {"cmd": "get_action", "instruction": instruction}, obs)
        header, arrays = recv_msg(self.sock)
        if header.get("error"):
            raise RuntimeError(f"GR00T server error: {header['error']}")
        return header, arrays

    def close(self) -> None:
        self.sock.close()
