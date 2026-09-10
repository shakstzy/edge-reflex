# edge-reflex

> **Sub-2ms Deterministic Reflex Shield for Vision-Language-Action (VLA) Robotics**  
> Built by Adithya ([@shakstzy](https://github.com/shakstzy)) at **Dipar**.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/target-Jetson%20%7C%20ARM%20%7C%20x86-orange.svg)]()

---

## The Problem: The 10 Hz Blind Window

Modern Vision-Language-Action (VLA) foundation models ($\\pi_0$, OpenVLA, Octo, RT-2) solve semantic generalization—they know what objects are and how to interact with unstructured environments.

**However, they are fatally slow for closed-loop physics:**
- Denoising a diffusion policy or running autoregressive transformer tokens takes **70 ms to 200 ms** per decision cycle.
- A 13 Hz control loop means the robot travels **11.5 cm completely blind** between decisions at standard human walking speed ($1.5\\text{ m/s}$).
- When an end-effector slips on wet oil, hits dynamic resistance, or receives an external impulse, a 13 Hz policy drops the payload or strips actuator gears before the GPU finishes denoising step 10.

Scaling up to 120W cluster GPUs on mobile robots drains battery payloads and still cannot beat algorithmic latency barriers.

---

## The Solution: A High-Speed Reflex Shield Underneath the VLA

`edge-reflex` is an ultra-lightweight, software-defined co-processor architecture that runs **underneath** any high-level VLA planner at **>800 Hz (<1.25 ms)** on standard edge CPUs (Nvidia Jetson Orin, ARM Neoverse, x86).

```
[ Camera / Sensors ]
         │
         ▼
[ Saliency Patch Pruner ] ──> Discards 70% of static background tokens before compute
         │
         ▼
[ JEPA Latent Predictor ] ──> Predicts physics in abstract latent space (R^64) without pixel diffusion
         │
         ▼
[ Vectorized MPPI Core ]  ──> Evaluates 256 parallel candidate rollouts in <1ms
         │
         ▼
[ Deterministic CBF Filter ] ──> Quadratic-barrier projection: guarantees torque & joint limits
         │
         ▼
[ CAN-FD / EtherCAT Motors ] (Sub-2ms reaction to contact slips)
```

1. **70% Spatial-Temporal Patch Pruning**: Computes temporal and optical gradient energy across image patches, discarding static tables, floors, and background walls before matrix multiplication.
2. **JEPA Latent Forward Dynamics**: Operates in low-dimensional abstract embedding space ($z \\in \\mathbb{R}^{64}$) rather than autoregressively generating RGB pixels.
3. **Parallel Vectorized MPPI**: Simulates 256 candidate perturbation rollouts simultaneously with zero thread contention.
4. **Deterministic Control Barrier Functions (CBF)**: Quadratic-barrier filter mathematically bounding joint torques ($L_f h + L_g h \\cdot u + \\gamma h \\ge 0$), preventing crashes even if the upstream neural network hallucinates.

---

## Empirical Benchmark

Run on off-the-shelf single-core CPU (dt = 3.2 ms simulation step, 40 N·m external shock perturbation):

| Performance Metric | Diffusion Policy ($\\pi_0$) | Autoregressive VLA (RT-2) | **Edge Reflex (Ours)** |
| :--- | :---: | :---: | :---: |
| **Decision Latency** | 77.0 ms | 205.0 ms | **1.24 ms** |
| **Control Frequency** | 13.0 Hz | 4.9 Hz | **805.8 Hz** |
| **SRAM Token Pruning** | 0% (Full Image) | 0% (Full Image) | **70% Pruned** |
| **Shock Reaction Lag** | 77.0 ms | 205.0 ms | **1.24 ms** |
| **Max Slip Deflection** | 4.70 cm | 33.60 cm | **4.57 cm** |
| **Kinematic Safety** | None (Black box) | None (Black box) | **100% CBF Bound** |
| **Payload Outcome** | ❌ **DROPPED** (Slip) | ❌ **CRASH** (Overheat) | ✅ **SAVED** (Recovered) |

---

## Quickstart

### 1. Clone & Run the Self-Contained Benchmark

Zero heavy dependencies required—pure, vectorized NumPy:

```bash
git clone https://github.com/shakstzy/edge-reflex.git
cd edge-reflex
python3 edge_reflex.py
```

### 2. Output

```
====================================================================================
      JEPA EDGE REFLEX CO-PROCESSOR: REAL PHYSICAL SYSTEM DEMONSTRATION
      Dual-Link Robotics Manipulator | 70% Token Pruning | CBF Safety Filter
====================================================================================

[*] Executing 50-step closed loop at 312.5 Hz (dt = 3.2ms)...
[*] ADVERSARIAL STRESS TEST: Injecting 40 N*m shock impulse at Step 10 (t = 32ms)...

====================================================================================
                  PHYSICAL AI DISTURBANCE & LATENCY REALITY
====================================================================================
Performance Metric               | Diffusion (pi0) | AutoReg (RT-2)  | JEPA Reflex (Ours)
------------------------------------------------------------------------------------
Decision Latency                 |         77.0 ms |        205.0 ms |         1.24 ms
Reflex Loop Frequency            |         13.0 Hz |          4.9 Hz |        805.8 Hz
SRAM Token Pruning               | 0% (Full Image) | 0% (Full Image) |      70% Pruned
Shock Blind Lag                  |         77.0 ms |        205.0 ms |         1.24 ms
Max Slip Deflection              |          4.7 cm |         33.6 cm |         4.57 cm
Kinematic Safety Guarantee       | None (Hallucinate) | None (Hallucinate) |  100% CBF Bound
Payload Outcome                  |  DROPPED (Slip) | CRASH (Overheat) | SAVED (Recovered)
====================================================================================
[*] REACTION SPEEDUP:     62.0x lower reflex latency (1.24ms vs 77.0ms)
[*] SLIP DEFLECTION:      4.57 cm vs 4.7 cm (payload saved)
[*] CBF CERTIFICATE:      100% of motor actions bounded within joint torque envelope
====================================================================================
```

---

## Architectural Context & Prior Art

`edge-reflex` integrates classical control theory with modern physical AI architectures:
- **Control Barrier Functions**: Ames et al., IEEE TAC 2014 (*"Control Barrier Function Based Quadratic Programs for Safety Critical Systems"*)
- **Model Predictive Path Integral (MPPI)**: Theodorou et al., IEEE CDC 2015 (*"Model Predictive Path Integral Control from a Stochastic HJB Perspective"*)
- **Joint Embedding Predictive Architectures (JEPA)**: LeCun et al., Meta FAIR 2022 (*"A Path Towards Autonomous Machine Intelligence"*)
- **Vision Token Pruning**: Bolya et al., Meta 2022 (*"ToMe: Token Merging for Fast Vision Transformers"*)

---

## Roadmap

- [x] Pure NumPy dual-link rigid body dynamics & MPPI simulation harness
- [x] Saliency patch pruner & quadratic CBF barrier projection
- [ ] Drop-in ROS 2 / Zenoh C++ node (`rclcpp`)
- [ ] Hugging Face LeRobot integration wrapper
- [ ] ISO 13849 / IEC 61508 formal verification test suite for industrial safety compliance

---

## License

Apache License 2.0. Copyright (c) 2026 Adithya ([@shakstzy](https://github.com/shakstzy)) / Dipar.
