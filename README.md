Using Isaac Sim for the ROV Reinforcement Learning Digital Twin
This project explores the integration of NVIDIA Isaac Sim with an existing ROV reinforcement learning (RL) digital twin. The framework combines marine vehicle dynamics, physics-informed learning and reinforcement learning to investigate autonomous ROV control and pipeline-tracking applications.
The implementation is designed for GPU-accelerated simulation and training on NVIDIA hardware.
> **Note:** This README provides an overview of the workflow and environment setup. Certain implementation details, model parameters, training configurations, simulation assets and integration interfaces have been intentionally omitted.
---
1. System Overview
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
Isaac Lab provides the reinforcement learning layer on top of Isaac Sim (Section 9), and an AI agent supervises training and evaluation runs (Section 10).
---
2. Hardware and Software
The development workflow uses:
NVIDIA GPU computing platform
ARM64/Grace-based computing environment
NVIDIA Isaac Sim
NVIDIA Isaac Lab (parallel-environment reinforcement learning)
Docker-based deployment
Python
PPO reinforcement learning
LLM-based supervisor agent (locally hosted or API-hosted model)
6-DOF marine vehicle dynamics
Physics-informed neural networks (PINNs)
USD-based simulation assets
Specific hardware configurations, software versions and dependency combinations are intentionally not disclosed in this public README.
---
3. Isaac Sim Environment
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
4. Initial Verification
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
5. ROV Digital Twin
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
6. Physics-Informed Learning
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
Hydrodynamic parameter estimation
Model correction
Uncertainty representation
Disturbance modelling
Adaptive simulation
The network architecture, training parameters, loss-function formulation and calibrated parameters are implementation-specific and are therefore omitted.
---
7. Reinforcement Learning
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
8. Pipeline-Tracking Application
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
9. Using Isaac Lab
Isaac Lab is NVIDIA's open-source robot-learning framework, built on top of Isaac Sim. Isaac Sim supplies the physics, the USD scene and the rendering. Isaac Lab adds the reinforcement learning layer: environment definitions, many copies of the environment simulated in parallel on the GPU, and ready-made training and evaluation scripts. In this project it is used to train and evaluate the pipeline-tracking controller at scale.
```text
Isaac Sim (physics, USD scene, rendering)
        │
        ▼
Isaac Lab (RL environments, parallel simulation)
        │
        ▼
Training Scripts (PPO and other algorithms)
        │
        ▼
Policy Checkpoint
        │
        ▼
Evaluation / Playback in Isaac Sim
```
9.1 Setup
Isaac Sim should already be verified (Section 4). Isaac Lab is then installed against that Isaac Sim installation:
```bash
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# point Isaac Lab to the existing Isaac Sim installation
ln -sfn <ISAACSIM_PATH> _isaac_sim

# install Isaac Lab and its extensions
./isaaclab.sh --install
```
The Isaac Sim build must match the machine architecture (for example, an aarch64 build on an ARM64/Grace system).
9.2 Check the installation
List the registered environments to confirm that Isaac Lab starts correctly:
```bash
./isaaclab.sh -p scripts/environments/list_envs.py
```
Always run Isaac Lab scripts through `./isaaclab.sh -p`, so that they use the Python environment that Isaac Sim provides.
9.3 Train a policy
```bash
./isaaclab.sh -p scripts/reinforcement_learning/<FRAMEWORK>/train.py \
    --task <TASK_NAME> --headless --num_envs <N>
```
`--headless` turns off the viewport, which makes training much faster.
`--num_envs` sets how many environments are simulated in parallel on the GPU.
Training progress and checkpoints are written to a logs folder for each run.
9.4 Evaluate a checkpoint
```bash
./isaaclab.sh -p scripts/reinforcement_learning/<FRAMEWORK>/play.py \
    --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```
Evaluation replays a trained policy without learning. Run it headless to collect metrics, or with the viewport to watch the ROV follow the pipeline.
Script locations and argument names change between Isaac Lab versions, so check the documentation for the installed version.
9.5 Running from the dashboard
The project dashboard exposes training and evaluation as Start, Stop and Restart controls, with selectors for the algorithm and checkpoint. The same workflow can therefore be run without typing commands, and the AI agent in Section 10 uses the same controls.
Task names, environment configurations, parallel-environment counts and training settings are implementation-specific and are omitted.
---
10. AI Agent Integration
10.1 Purpose
Training an RL controller involves many repeated manual decisions: stopping a run that has stopped improving, restarting from a better checkpoint, or retuning a parameter. The AI agent automates these decisions. It is a large language model (LLM) that watches training and evaluation runs through the dashboard and decides what to do next.
Its purposes are to:
Detect problems early, such as a reward plateau, divergence, NaN values, a crashed process or GPU memory exhaustion.
Save GPU time by stopping unproductive runs and restarting from the best checkpoint.
Make decisions consistent and traceable, with the evidence, the action and the outcome recorded for each one.
Free the researcher from watching long runs, and ask for help when the situation is unclear.
A second, planned role is a mission agent that accepts a plain-language mission (for example, "inspect the full pipeline and report any rock close to it") and calls a fixed set of predefined skills.
10.2 Where the agent sits
The agent does not control the ROV directly. Two loops are kept strictly separate:
```text
FAST LOOP (physics rate, no LLM)
   RL Policy ──► Thruster Commands ──► Isaac Sim ──► New State
        ▲                                               │
        └───────────────────────────────────────────────┘

SLOW LOOP (every few seconds or on events)
   Training / Evaluation Runs
              │  metrics, status, logs
              ▼
      Dashboard (control interface)
              │  compact summary, not raw logs
              ▼
        AI Agent (LLM)
              │  proposed action
              ▼
   Guardrails (deterministic code)
              │  validated action
              ▼
   Dashboard controls ──► Stop / Restart / Switch Checkpoint / Retune
```
The LLM is far too slow for thruster control, so the RL policy keeps real-time control. The agent only makes slower, high-level decisions about the runs.
10.3 What the agent can and cannot do
The agent can choose from a fixed, typed set of actions:
Continue the current run.
Stop a run, with a stated reason.
Restart a run from a checkpoint, with optional configuration changes.
Switch to another checkpoint, chosen by evaluation metrics rather than simply the latest.
Retune a parameter, but only parameters on an allowlist and only within set minimum and maximum values.
Run an evaluation of a checkpoint.
Escalate a question to the researcher.
The agent cannot control thrusters, delete checkpoints, change parameters outside the allowed bounds, or run any command that is not on the fixed allowlist.
10.4 Safety
Every proposed action is checked by deterministic code, not by the LLM, against the allowlist, the parameter bounds and a locked file of hard limits that the agent cannot change.
Limits apply to restarts per hour, total retunes, run time and GPU budget, with a cooldown after each intervention so that its effect can be measured.
The agent does not repeat an intervention that has already failed, and escalates to the researcher after repeated failures.
If the LLM is unavailable or returns an invalid action, nothing is done and the event is logged. Stopping a healthy run by mistake is considered worse than waiting.
The agent runs in one of three modes: observe-only (it only logs what it would do), approve-each (the researcher approves every action) or autonomous. The default is observe-only.
Every decision is written to a log with the evidence used, the action taken and the result.
10.5 Settings
A settings page in the dashboard lets the researcher change the agent's behaviour without editing files: the mode, how often it checks, what counts as a plateau or collapse, which actions are permitted, the allowed range of each tunable parameter, the limits, the model used (local or API-hosted) and the checkpoint rules. Conservative, balanced and aggressive presets are provided, every change is versioned and can be undone, and a dry run shows what the agent would have done with new settings before they are applied.
10.6 Status and evaluation
The agent is under development. It is first run in observe-only mode and its decisions are compared with the manual interventions made in earlier runs. It is also tested with injected faults (a reward plateau, NaN loss, a crashed process, GPU memory exhaustion and a checkpoint that evaluates worse than an older one). The measures of interest are the rate of correct interventions, the number of wrongly stopped runs (the target is zero), the time taken to detect a problem and the GPU hours saved compared with manual supervision.
Prompts, thresholds, limits, model choices and the decision logic are implementation-specific and are omitted.
---
11. Development Workflow
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
10. Install Isaac Lab and check the environment list
          ↓
11. Configure PPO training
          ↓
12. Train controller headless with Isaac Lab
          ↓
13. Evaluate ROV performance
          ↓
14. Enable the AI agent in observe-only mode
          ↓
15. Review its decisions, then move to approve-each
```
---
12. Research Objective
The overall objective is to establish a GPU-accelerated digital twin framework for investigating:
Autonomous ROV control
Reinforcement learning
LLM-based supervision of training and evaluation
Physics-informed learning
Pipeline tracking
Robust control under uncertainty
Sim-to-real transfer
Intelligent subsea robotics
