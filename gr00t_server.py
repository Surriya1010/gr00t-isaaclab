"""GR00T N1 policy server for the Isaac Lab GR1-T2 closed-loop tasks. Run with the GR00T venv:

    /path/to/Isaac-GR00T/.venv/bin/python gr00t_server.py \
        --model_path ".../checkpoints/GR00T-N1-2B-tuned-Nut-Pouring-task"

Listens on 127.0.0.1 only. Image preprocessing matches IsaacLabEvalTasks: the 160x256 head-camera
frame is zero-padded top/bottom to 256x256.
"""

import argparse
import os
import socket
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bridge import recv_msg, send_msg  # noqa: E402

from gr00t.experiment.data_config import DATA_CONFIG_MAP  # noqa: E402
from gr00t.model.policy import Gr00tPolicy  # noqa: E402

GROUPS = ["left_arm", "right_arm", "left_hand", "right_hand"]

p = argparse.ArgumentParser()
p.add_argument("--model_path", required=True)
p.add_argument("--port", type=int, default=5566)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
p.add_argument("--denoising_steps", type=int, default=4)
args = p.parse_args()

cfg = DATA_CONFIG_MAP["gr1_arms_only"]
policy = Gr00tPolicy(
    model_path=args.model_path,
    embodiment_tag="gr1",
    modality_config=cfg.modality_config(),
    modality_transform=cfg.transform(),
    denoising_steps=args.denoising_steps,
    device=args.device,
)
print(f"[gr00t_server] loaded {args.model_path} on {args.device}", flush=True)


def pad_to_square(rgb: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) with H < W -> (N, W, W, 3), zero padding split top/bottom."""
    pad = (rgb.shape[2] - rgb.shape[1]) // 2
    return np.pad(rgb, ((0, 0), (pad, rgb.shape[2] - rgb.shape[1] - pad), (0, 0), (0, 0)))


def handle(instruction: str, obs: dict) -> dict:
    rgb = pad_to_square(obs["rgb"].astype(np.uint8))  # (N, 256, 256, 3)
    n = rgb.shape[0]
    policy_obs = {
        "annotation.human.action.task_description": [instruction] * n,
        "video.ego_view": rgb.reshape(n, 1, *rgb.shape[1:]),
    }
    for g in GROUPS:
        policy_obs[f"state.{g}"] = obs[g].astype(np.float64).reshape(n, 1, -1)
    action = policy.get_action(policy_obs)
    return {g: np.asarray(action[f"action.{g}"], dtype=np.float32) for g in GROUPS}  # (N, 16, D)


srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", args.port))
srv.listen(1)
print(f"[gr00t_server] ready on 127.0.0.1:{args.port}", flush=True)

while True:
    conn, _ = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    print("[gr00t_server] client connected", flush=True)
    try:
        while True:
            header, arrays = recv_msg(conn)
            if header.get("cmd") == "ping":
                send_msg(conn, {"ok": True, "model": os.path.basename(args.model_path.rstrip("/"))})
                continue
            t0 = time.perf_counter()
            try:
                out = handle(header["instruction"], arrays)
                send_msg(conn, {"infer_ms": round((time.perf_counter() - t0) * 1000, 1)}, out)
            except Exception as e:  # report to the client instead of dying
                send_msg(conn, {"error": f"{type(e).__name__}: {e}"})
    except ConnectionError:
        print("[gr00t_server] client disconnected", flush=True)
    finally:
        conn.close()
