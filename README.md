# GR00T N1 closed-loop in Isaac Lab 3.0

Runs NVIDIA's **GR00T N1 (2B)** vision-language-action model closed-loop on the
**GR1-T2 humanoid** manipulation tasks in **Isaac Lab 3.0**. The point is to evaluate a
released checkpoint by acting in simulation, instead of only scoring it offline.

> **Status:** integration only. The loop runs end to end. No success-rate results are
> published here yet.

## Why a bridge
Isaac Lab 3.0 runs on Python 3.12 and GR00T N1 needs its own Python 3.10 environment, so
they can't share one process. This repo splits them in two:

```
Isaac Lab (py3.12)                          GR00T N1 server (py3.10)
run_isaac_gr00t.py  --obs (JSON + .npz)-->  gr00t_server.py
  GR1-T2 task env   <--16-step action chunk--  policy.get_action()
```

- `bridge.py`: a minimal localhost protocol. Each message is an 8-byte length header,
  then JSON, then `.npz` arrays loaded with `allow_pickle=False`, so no code can travel
  over the socket. It uses only the standard library and numpy, so it imports in both
  environments.
- `gr00t_server.py`: loads a GR00T N1 checkpoint and serves action chunks. The
  160×256 head-camera frame is zero-padded to 256×256 to match the reference
  IsaacLabEvalTasks preprocessing.
- `gr1_tasks.py`: registers closed-loop variants of the GR1-T2 tasks (for example
  `Isaac-NutPour-GR1T2-ClosedLoop-v0` and `Isaac-ExhaustPipe-GR1T2-ClosedLoop-v0`).
- `run_isaac_gr00t.py`: the episode runner. It has a **`--policy hold`** baseline that
  keeps the robot still, so the simulator's own cost can be measured before any
  slowdown is blamed on model inference.

## Run
```bash
# 1) policy server, in the GR00T venv
/path/to/Isaac-GR00T/.venv/bin/python gr00t_server.py --model_path <checkpoint>

# 2) simulator baseline (no model)
cd ~/IsaacLab
./isaaclab.sh -p /path/to/gr00t-isaaclab/run_isaac_gr00t.py --policy hold

# 3) closed loop
./isaaclab.sh -p /path/to/gr00t-isaaclab/run_isaac_gr00t.py --policy gr00t \
    --task Isaac-NutPour-GR1T2-ClosedLoop-v0 --episodes 3 --chunks 30
```
Useful flags: `--headless`, `--device cpu|cuda:0`, `--rendering_mode performance|balanced|quality`,
`--denoising_steps` (server), `--feedback_actions` (steps executed per 16-step chunk).

## Requirements
- Isaac Sim / Isaac Lab 3.0
- [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T), installed in its own venv, plus a
  GR00T N1 checkpoint

## Next
- Report success rate over fixed seeds, with confidence intervals, per task
- Measure how chunk length (`--feedback_actions`) trades off reactivity against success

GR00T, Isaac Sim and Isaac Lab are NVIDIA projects under their own licences. This repo
contains only the integration code.
