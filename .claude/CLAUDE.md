# Workspace Overview

This VS Code workspace (`UWLab.code-workspace`) contains three repos:

- **UWLab** (`/home/joshuat26/Documents/Github/UWLab`) — A robotics simulation framework built on Isaac Lab for robot learning tasks (manipulation, locomotion, etc.) developed at UW.
- **rsl_rl** (`/home/joshuat26/Documents/Github/rsl_rl`) — A fast, simple reinforcement learning library for robotics (from RSL/ETH Zurich).
- **obsidian-vault** (`/home/joshuat26/Documents/Github/obsidian-vault`) — A personal notes/knowledge base (Obsidian markdown collection).

# Environment Setup

To run UWLab code, activate the conda environment:

```bash
conda activate env_uwlab
```

# Active Project: FastSAC on Peg Insertion (State-Based)

**Goal:** Get FastSACAgent (holosoma) working on the UR5e peg insertion task from state observations. Implementation is complete — focus is now on training, debugging, and tuning.

**Tasks:** `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-OffPolicy-v0` (full), `OmniReset-Ur5eRobotiq2f85-RelCartesianOSC-State-Easy-OffPolicy-v0` (easier variant).

**Working preference:** Only implement what is explicitly asked for in each step. Do not jump ahead or implement features preemptively.
