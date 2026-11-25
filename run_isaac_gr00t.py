"""Run the GR1-T2 humanoid in Isaac Sim / Isaac Lab 3.0, driven by GR00T N1 (or held still to benchmark).

    cd ~/IsaacLab
    # 1) benchmark the sim alone (robot holds its pose):
    ./isaaclab.sh -p /path/to/gr00t-isaaclab/run_isaac_gr00t.py --policy hold
    # 2) closed loop with GR00T (start gr00t_server.py first):
    ./isaaclab.sh -p /path/to/gr00t-isaaclab/run_isaac_gr00t.py --policy gr00t

Useful flags: --task Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0, --episodes N, --chunks N (16 steps each),
--device cpu|cuda:0 (physics), --headless, --rendering_mode performance|balanced|quality.
"""

import argparse
import contextlib
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401,E402
from isaaclab_tasks.utils import add_launcher_args, launch_simulation, resolve_task_config, setup_preset_cli  # noqa: E402

import gr1_tasks  # noqa: E402  (registers the ClosedLoop tasks)

parser = argparse.ArgumentParser(description="GR00T N1 x Isaac Lab GR1-T2 closed loop")
parser.add_argument("--task", default="Isaac-NutPour-GR1T2-ClosedLoop-v0")
parser.add_argument("--policy", choices=["hold", "gr00t"], default="hold")
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--chunks", type=int, default=30, help="action chunks per episode (16 steps each)")
parser.add_argument("--feedback_actions", type=int, default=16, help="steps executed per chunk")
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--server_port", type=int, default=5566)
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"], enable_cameras=True, rendering_mode="performance")
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args


def to_torch(x):
    try:
        import warp as wp

        if isinstance(x, wp.array):
            return wp.to_torch(x)
    except ImportError:
        pass
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def gpu_mem():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        used, total, util = [s.strip() for s in out.split(",")]
        return f"VRAM {used}/{total} MiB, GPU util {util}%"
    except Exception:
        return "VRAM n/a"


def main():
    torch.manual_seed(args_cli.seed)
    env_cfg, _ = resolve_task_config(args_cli.task, "")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = 1
        if args_cli.device is not None:
            env_cfg.sim.device = args_cli.device
        success_term = env_cfg.terminations.success
        env_cfg.terminations = {}  # episodes are handled here, no auto-reset mid-rollout

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        sim = env.sim
        dev = env.device
        robot = env.scene["robot"]
        cam = env.scene["robot_pov_cam"]

        names = list(getattr(robot, "joint_names", None) or robot.data.joint_names)
        idx = {n: i for i, n in enumerate(names)}
        action_ids = torch.tensor([idx[n] for n in gr1_tasks.ACTION_JOINTS], device=dev)
        group_ids = {g: [idx[n] for n in js] for g, js in gr1_tasks.GR00T_GROUPS.items()}
        term = env.action_manager.get_term("gr1_action")
        resolved = list(getattr(term, "_joint_names", gr1_tasks.ACTION_JOINTS))
        assert resolved == gr1_tasks.ACTION_JOINTS, f"action joint order mismatch: {resolved[:6]}..."
        print(f"[run] {args_cli.task} | physics device {env_cfg.sim.device} | {len(names)} robot joints, "
              f"action dim {env.action_space.shape[-1]} | policy={args_cli.policy}", flush=True)

        client = None
        instruction = gr1_tasks.INSTRUCTIONS[args_cli.task]
        if args_cli.policy == "gr00t":
            from bridge import Gr00tClient

            client = Gr00tClient(port=args_cli.server_port)
            print(f"[run] GR00T server: {client.ping()} | instruction: {instruction}", flush=True)

        def running():
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                return False
            return True

        sim_dt = env_cfg.sim.dt * env_cfg.decimation
        results = []
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            for ep in range(args_cli.episodes):
                if not running():
                    break
                env.reset(seed=args_cli.seed + ep)
                steps, t_ep, infer_ms, success = 0, time.perf_counter(), [], False
                t_win, s_win = time.perf_counter(), 0
                for c in range(args_cli.chunks):
                    if not running():
                        break
                    q = to_torch(robot.data.joint_pos)
                    if client is None:
                        chunk = q[:, action_ids].unsqueeze(1).repeat(1, args_cli.feedback_actions, 1)
                    else:
                        rgb = to_torch(cam.data.output["rgb"])[..., :3].to(torch.uint8).cpu().numpy()
                        qn = q.cpu().numpy()
                        obs = {"rgb": rgb, **{g: qn[:, ids] for g, ids in group_ids.items()}}
                        hdr, act = client.get_action(instruction, obs)
                        infer_ms.append(hdr["infer_ms"])
                        # policy groups -> 36-dim sim action; joints GR00T doesn't predict stay 0 (as in
                        # IsaacLabEvalTasks: the hand mimic joints).
                        horizon = act["left_arm"].shape[1]
                        chunk = torch.zeros((1, horizon, len(gr1_tasks.ACTION_JOINTS)), device=dev)
                        for g, js in gr1_tasks.GR00T_GROUPS.items():
                            for k, n in enumerate(js):
                                chunk[..., gr1_tasks.ACTION_JOINTS.index(n)] = torch.from_numpy(act[g][..., k]).to(dev)
                    for i in range(min(args_cli.feedback_actions, chunk.shape[1])):
                        env.step(chunk[:, i, :])
                        steps += 1
                        s_win += 1
                    success = success or bool(to_torch(success_term.func(env, **(success_term.params or {})))[0])
                    now = time.perf_counter()
                    if now - t_win > 5 or c == args_cli.chunks - 1:
                        sps = s_win / (now - t_win)
                        print(f"  ep {ep} chunk {c + 1}/{args_cli.chunks}: {sps:.1f} steps/s "
                              f"(real-time x{sps * sim_dt:.2f})"
                              + (f", GR00T {np.mean(infer_ms[-5:]):.0f} ms/chunk" if infer_ms else "")
                              + f", success={success} | {gpu_mem()}", flush=True)
                        t_win, s_win = now, 0
                    if success:
                        break
                wall = time.perf_counter() - t_ep
                results.append(success)
                print(f"[run] episode {ep}: success={success}, {steps} steps in {wall:.0f}s "
                      f"(sim {steps * sim_dt:.0f}s)", flush=True)
        if results:
            print(f"[run] success rate {np.mean(results):.0%} over {len(results)} episode(s)", flush=True)
        if client is not None:
            client.close()
        env.close()


if __name__ == "__main__":
    main()
