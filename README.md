# Using Isaac Sim for the ROV Reinforcement Learning Digital Twin

This project explores the integration of **NVIDIA Isaac Sim** with an existing **ROV reinforcement learning (RL) digital twin**. The framework combines marine vehicle dynamics, physics-informed learning and reinforcement learning to investigate autonomous ROV control and pipeline-tracking applications.

The implementation is designed for GPU-accelerated simulation and training on NVIDIA hardware.

> **Note:** This README provides an overview of the workflow and environment setup. Certain implementation details, model parameters, training configurations, simulation assets and integration interfaces have been intentionally omitted.

---

## 1. System Overview

The digital twin combines several components:

```text
        ROV Hydrodynamic Model
                 │
                 ▼
        Physics-Informed Model
                 │
                 ▼
          ROV Digital Twin
                 │
                 ▼
           NVIDIA Isaac Sim
                 │
                 ▼
          RL Environment
                 │
                 ▼
             PPO Agent
                 │
                 ▼
       Autonomous ROV Control
```

Isaac Sim provides the simulation environment, while the existing marine dynamics and learning components provide the basis for ROV behaviour and control.

---

## 2. Hardware and Software

The development workflow uses:

* NVIDIA GPU computing platform
* ARM64/Grace-based computing environment
* NVIDIA Isaac Sim
* Docker-based deployment
* Python
* PPO reinforcement learning
* 6-DOF marine vehicle dynamics
* Physics-informed neural networks (PINNs)
* USD-based simulation assets

Specific hardware configurations, software versions and dependency combinations are intentionally not disclosed in this public README.

---

## 3. Isaac Sim Environment

Isaac Sim is deployed using an NVIDIA-supported container appropriate for the target GPU architecture.

A generic container workflow is:

```bash
docker pull nvcr.io/nvidia/isaac-sim:<VERSION>-<ARCH>
```

The container is then launched with GPU access and a project workspace:

```bash
docker run --gpus all -it --rm \
    -v <PROJECT_DIRECTORY>:/workspace \
    nvcr.io/nvidia/isaac-sim:<VERSION>-<ARCH>
```

The exact image version, architecture tag and project directory used in the implementation are intentionally omitted.

---

## 4. Initial Verification

Before integrating the ROV digital twin, Isaac Sim should be verified independently.

For a headless environment, the corresponding Isaac Sim headless launcher can be used:

```bash
./isaac-sim.headless.sh
```

The initial verification should confirm:

```text
GPU
 │
 ▼
Container
 │
 ▼
Isaac Sim
 │
 ▼
Headless Rendering
 │
 ▼
Simulation
```

Only after this stage has been successfully completed should the ROV environment be introduced.

---

## 5. ROV Digital Twin

The ROV digital twin incorporates a six-degree-of-freedom marine vehicle representation.

The general state consists of position, orientation, linear velocity and angular velocity:

```text
ROV State
   │
   ├── Position
   ├── Orientation
   ├── Linear Velocity
   └── Angular Velocity
```

The underlying hydrodynamic formulation follows established marine vehicle modelling approaches.

Specific hydrodynamic coefficients, vehicle dimensions, mass properties, damping parameters and other calibrated model parameters are not included.

---

## 6. Physics-Informed Learning

A physics-informed neural network is incorporated to represent selected aspects of the ROV's uncertain or nonlinear behaviour.

The general concept is:

```text
Simulation / Experimental Data
             │
             ▼
       Neural Network
             │
             ▼
      Physics Constraints
             │
             ▼
     Physics-Informed Model
```

The PINN can support applications such as:

* Hydrodynamic parameter estimation
* Model correction
* Uncertainty representation
* Disturbance modelling
* Adaptive simulation

The network architecture, training parameters, loss-function formulation and calibrated parameters are implementation-specific and are therefore omitted.

---

## 7. Reinforcement Learning

The RL component uses PPO to learn an ROV control policy.

The general training loop is:

```text
ROV Observation
       │
       ▼
    PPO Policy
       │
       ▼
 Control Action
       │
       ▼
   Isaac Sim
       │
       ▼
   New State
       │
       ▼
    Reward
       │
       └──────────► PPO Update
```

The policy is trained to achieve the required ROV control objective while maintaining stable and physically meaningful behaviour.

Exact observation definitions, action mappings, network architecture, PPO hyperparameters and reward coefficients are not disclosed.

---

## 8. Pipeline-Tracking Application

One application considered by the digital twin is autonomous underwater pipeline tracking.

The general objective is:

```text
Reference Pipeline
        │
        ▼
Tracking Error
        │
        ▼
   PPO Controller
        │
        ▼
    ROV Motion
        │
        ▼
Tracking Performance
```

The simulation can incorporate environmental disturbances and sensor uncertainty to evaluate the robustness of the learned controller.

The precise pipeline geometry, sensor configuration, tracking thresholds and reward formulation are implementation-specific.

---

## 9. Development Workflow

The recommended development sequence is:

```text
1. Configure GPU environment
          ↓
2. Verify container GPU access
          ↓
3. Start Isaac Sim
          ↓
4. Verify headless simulation
          ↓
5. Import ROV simulation asset
          ↓
6. Configure sensors
          ↓
7. Connect vehicle dynamics
          ↓
8. Integrate physics-informed model
          ↓
9. Create RL environment
          ↓
10. Configure PPO training
          ↓
11. Train controller
          ↓
12. Evaluate ROV performance
```

---

## 10. Research Objective

The overall objective is to establish a GPU-accelerated digital twin framework for investigating:

* Autonomous ROV control
* Reinforcement learning
* Physics-informed learning
* Pipeline tracking
* Robust control under uncertainty
* Sim-to-real transfer
* Intelligent subsea robotics


=======================

# rov_pipeline_v20

ROV pipeline-tracking simulation, training and evaluation on **Isaac Lab** (GPU-resident PhysX),
the step-3+ result of `V18_ISAAC_LAB_PLAN.md`'s migration off v19's numpy/RK4 proxy env. Eleven
control methods share one environment and one action interface: 5 learned RL policies (PPO,
TD3, DDPG, SAC, TRPO) and 6 classical/model-based controllers (PID, SMC, Fuzzy, Feedback
Linearization, MPC, ANFIS).

A separate project, `../rov_dashboard_v1`, provides a point-and-click web UI over everything in
this README (plus live telemetry visualization and an optional LLM supervisor agent in
`../rov_agent_v1`) — see `rov_dashboard_v1/README.md` if you'd rather not use the terminal.

## Environment setup

```bash
source /home/mcschin1/env_isaaclab/bin/activate
export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
cd /home/mcschin1/Downloads/IsaacLab
```

Every script below is then launched the same way:
```bash
./isaaclab.sh -p /home/mcschin1/Downloads/nvidia_rov/rov_pipeline_v20/<script>.py [args] --headless
```
Drop `--headless` (or pass `--viz kit` where the script supports it) to watch it in Isaac Sim's
own Kit viewer instead.

## The environment

**`rov_pipeline_env_v20.py`** — `RovPipelineEnv`, a `DirectRLEnv` (GPU torch tensors, PhysX
integration). Observation encoding, reward terms and termination conditions are a 1:1 port of
v19's `PipelineTrackingEnv` — only the integration backend changed. `SCENE` (module-level dict)
holds the pipeline centerline points and rock positions/radii; `env._last` (populated every step
by `_get_dones()`) carries the full per-step state (`pos`, `roll/pitch/yaw`, `nu_lin`/`nu_ang`,
`s` (along-track progress), `cross_h`/`cross_h_signed` (cross-track error), `dist_to_pipe`,
`rock_dist`/`rock_bearing`, etc.) that every other script in this project reads from.

`test_rov_env_v20.py` is the standalone smoke test — launches the app, then imports the env
module (the standard Isaac Lab import-order constraint: `isaaclab.*` submodules can't be
imported before Kit's extension system is running, which is why every script here calls
`AppLauncher` before any other project import).

## The 11 methods

| Method | Checkpoint / class | Trainer (if learned) |
|---|---|---|
| `ppo` | `rov_pipeline_policy_v20.pt` | `rov_pipeline_rl_v20.py` |
| `td3` | `rov_pipeline_policy_td3_v20.pt` | `rov_pipeline_td3_v20.py` |
| `ddpg` | `rov_pipeline_policy_ddpg_v20.pt` | `rov_pipeline_ddpg_v20.py` |
| `sac` | `rov_pipeline_policy_sac_v20.pt` | `rov_pipeline_sac_v20.py` |
| `trpo` | `rov_pipeline_policy_trpo_v20.pt` | `rov_pipeline_trpo_v20.py` |
| `pid`, `smc`, `fuzzy`, `fl` | classes in `classical_controllers_v20.py` | — (no training) |
| `mpc` | `MPCControllerV20` (wraps v19's unmodified `MPCController`) | — |
| `anfis` | `ANFISController` (loads `rov_pipeline_v19/anfis_channels_v19.pt`) | — |

All 5 RL trainers share `ppo_v20.py`'s `ActorCritic` network/checkpoint format (PPO and TRPO are
both on-policy and use it directly; `off_policy_v20.py` is TD3/DDPG/SAC's shared replay-buffer
infrastructure, a torch port of v19's `off_policy_common_v19.py`). Classical controllers are a
torch port of v19's `classical_controller_common_v19.py` — same gains, same control laws, only
the tensor backend and state-access paths changed (`env._project_to_centerline`/`env._robot.data`
instead of v19's numpy `env.state`/`env.seg_dir`).

**Training** (headless, 64 parallel envs):
```bash
./isaaclab.sh -p rov_pipeline_trpo_v20.py --total-timesteps 20000000 --headless
```
Each trainer's own `--help` lists its flags; commonly: `--total-timesteps`, `--n-envs`,
`--seed`, plus a few algorithm-specific knobs (PPO: `--ent-coef`; TRPO: `--gamma`,
`--action-penalty-coef`; TD3/DDPG/SAC: `--warmup-steps`, `--log-every-steps`). All of them
accept `--resume-from <checkpoint.pt>` (PPO/TRPO) to continue training a saved policy. Progress
is logged every update to `logs/<algo>/training_log.csv` (PPO logs to `logs/training_log.csv`
directly) with columns `update, timesteps, mean_ep_return, mean_ep_len, success_rate,
flip_rate, pipe_gap_violation_rate, policy_loss, value_loss, entropy, elapsed_s`.

`run_fixed_offpolicy_chain.sh` is the batch-run harness used to train TD3/DDPG/SAC/TRPO back to
back unattended: it restarts a run if no `step`/`update` log line appears within 240s of launch
— the known transient "Kit cold-start stall" — up to 3 attempts per algorithm before giving up.

**Deterministic evaluation** (why it exists: training-time rolling-average stats are noisy and
can be fooled into keeping a late, regressed checkpoint over an earlier better one — see the
dashboard/agent project's README for the specific historical incidents this caused):
```bash
./isaaclab.sh -p eval_policy_v20.py --checkpoint rov_pipeline_policy_trpo_v20.pt --headless
# classical controllers: run_classical_controller_v20.py --method mpc  (no --checkpoint)
```
Runs N full deterministic (no-sampling) episodes in parallel, reports success/flip/gap-violation
rates and standoff RMSE (`sqrt-mean-sq(dist_to_pipe - standoff_track_target)`, matching v19's
`setpoint_rmse` training-log column).

**Single-env GUI flight** (one episode, for visual inspection or screen recording):
```bash
./isaaclab.sh -p run_gui_flight_v20.py --method trpo --viz kit
```
Prints `Outcome: <success|flipped|pipe_gap_violation|other_collision|timed_out>` on episode end —
`record_gui_run_v20.sh` greps for that exact string to know when to stop recording. Also streams
live telemetry (position, depth, cross-track error, progress, speed, 8 thruster commands) over
UDP for `rov_dashboard_v1` to visualize, if that dashboard happens to be running — it's a no-op
if not, so this script works identically standalone.

## Trajectory recording and plots

```bash
./isaaclab.sh -p record_trajectory_v20.py --method <any-of-11> --headless
./isaaclab.sh -p plot_trajectories_v20.py          # reads logs/trajectories/<method>.npz
./isaaclab.sh -p make_plots_v20.py                 # training curves + 11-method comparison chart
```
`record_trajectory_v20.py` is kept separate from `run_gui_flight_v20.py` (rather than importing
it) because `run_gui_flight_v20.py` calls `AppLauncher` at import time, so it can't be imported
as a library once Kit is already running inside another script.

## Tuning sweeps

`sweep_mpc_v20.sh` — the MPC parameter sweep harness (`mpc_sweep.log` is its recorded output):
tries alternate CEM-sampling/horizon/weight configurations against the baseline over repeated
trials. In the run on record, every retuned variant scored at or below the untouched baseline,
so `classical_controllers_v20.py`'s `MPC_OVERRIDES` was deliberately left empty (`= {}`) —
MPC runs with its original parameters.

## Verifying physics before trusting the RL results

`verify_rov_physics_isaaclab_v20.py` — step 2 of the migration plan, no RL: loads the exact same
USD asset v19 used (`rov_in_water.usda`) into a minimal Isaac Lab env and checks the ROV
drifts/thrusts/settles the same way v17–v19's live GUI force applicator did, before any training
result from this version is trusted.

## Files

- `rov_pipeline_env_v20.py` — the shared environment (`RovPipelineEnv`, `SCENE`).
- `ppo_v20.py` — shared `ActorCritic` network (PPO + TRPO), GAE.
- `off_policy_v20.py`, `trpo_v20.py` — shared infra for TD3/DDPG/SAC and for TRPO respectively.
- `rov_pipeline_{rl,td3,ddpg,sac,trpo}_v20.py` — the 5 trainers (`rl` = PPO).
- `classical_controllers_v20.py` — PID/SMC/Fuzzy/FL/MPC/ANFIS, torch port of v19's controllers.
- `eval_policy_v20.py` / `run_classical_controller_v20.py` — deterministic N-episode evaluation
  for learned / classical methods respectively.
- `run_gui_flight_v20.py` — single-env one-episode flight, any of the 11 methods, GUI or headless.
- `record_trajectory_v20.py` / `plot_trajectories_v20.py` — trajectory capture and plotting.
- `make_plots_v20.py` — training-curve and 11-method comparison plots.
- `test_rov_env_v20.py` — environment smoke test.
- `verify_rov_physics_isaaclab_v20.py` — pre-RL physics sanity check against v19's model.
- `run_fixed_offpolicy_chain.sh` — unattended batch trainer with Kit-stall auto-restart.
- `record_gui_run_v20.sh` — screen-recording wrapper around `run_gui_flight_v20.py`.
- `sweep_mpc_v20.sh` — MPC parameter sweep harness.
- `rov_pipeline_policy*_v20.pt` — saved checkpoints (one per RL algorithm).
- `logs/` — `training_log.csv` per algorithm, `logs/trajectories/*.npz`.
- `plots/` — output of `make_plots_v20.py`/`plot_trajectories_v20.py`.
- `ROV_v20_User_Manual.docx` — longer-form manual (figures/screenshots) alongside this README.


 
