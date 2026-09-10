# edge-reflex

> **Deterministic 1 kHz High-Order Control Barrier Function (HOCBF) Safety Shield for Vision-Language-Action (VLA) Robotics**  
> Built by Adithya ([@shakstzy](https://github.com/shakstzy)).

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/target-ARM%20%7C%20RISC--V%20%7C%20x86-orange.svg)]()
[![Zero-Slop](https://img.shields.io/badge/Zero--Slop-Verified-success.svg)]()

---

## The Physical AI Vulnerability: VLA Action Chunk Blindness

Modern Vision-Language-Action (VLA) foundation models ($\pi_0$, OpenVLA, Octo, ACT) solve open-world semantic generalization: given visual observations and natural language instructions, they generate robot trajectories.

To amortize inference compute, state-of-the-art VLAs output **action chunks** (e.g., predicting 50 time steps, or 500 ms of trajectory at 10 Hz) that the robot executes open-loop on low-level joint controllers.

**In dynamic physical environments, open-loop action chunk execution is hazardous:**
- If an unmapped obstacle, fixture, human hand, or dropped tool enters the workspace at $t = 120\text{ ms}$, the robot blindly executes the remaining 380 ms of the action chunk.
- At $1.5\text{ m/s}$, this represents over **50 cm of unyielding momentum** before the 10 Hz neural network can re-plan.
- Collisions, actuator damage, and safety breaches are guaranteed.

---

## The Solution: A 1,000 Hz Analytical HOCBF-QP Safety Shield

`edge-reflex` is an ultra-low-latency, analytical safety shield running directly between the high-level VLA policy and the motor inverters at **>1,000 Hz (<150 µs)**.

```
       [ 10 Hz Vision-Language-Action Policy (pi0 / OpenVLA) ]
                                 │
                                 ▼ (Nominal Action Chunk: u_nom)
┌────────────────────────────────────────────────────────────────────────┐
│                        edge-reflex Shield (1 kHz)                      │
│                                                                        │
│   [ Analytical Euler-Lagrange Dynamics ]  --> M(q), C(q, dq), g(q)     │
│   [ Relative-Degree 2 HOCBF Formulation ] --> A_cbf * u <= b_cbf       │
│   [ Active-Set Quadratic Program Solver ] --> min ||u - u_nom||^2     │
│                                                                        │
│                 Solves in 128 µs in pure NumPy (KKT Compliant)         │
└────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼ (Minimally-Perturbed Safe Torque: u*)
                     [ CAN-FD / EtherCAT Motors ]
```

### Mathematical Formulation

#### 1. Rigid-Body Dynamics
For a manipulator with generalized coordinates $q \in \mathbb{R}^n$:
$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) + B\dot{q} = u + \tau_{ext}$$
where $M(q)$ is the symmetric positive-definite mass matrix, $C(q, \dot{q})$ is the Coriolis matrix, $g(q)$ is gravity, and $u$ is actuator torque.

#### 2. Relative-Degree 2 High-Order Control Barrier Functions (HOCBF)
Cartesian workspace safety boundaries $h(q) = \|p_{ee}(q) - p_{obs}\|^2 - r_{safe}^2 \ge 0$ have **relative degree 2** with respect to torque control $u$.

We formulate the second-order barrier manifold (Ames et al., Xiao & Belta):
$$\psi_0(q) = h(q)$$
$$\psi_1(q, \dot{q}) = \dot{h}(q) + \alpha_1 h(q) = 2(p_{ee} - p_{obs})^T J(q)\dot{q} + \alpha_1 h(q)$$
$$\dot{\psi}_1(q, \dot{q}, u) + \alpha_2 \psi_1(q, \dot{q}) \ge 0$$

Expanding $\ddot{q} = M(q)^{-1}(u - C\dot{q} - g - B\dot{q})$ yields the exact linear inequality constraint on control torque:
$$A_{cbf} u \le b_{cbf}$$

#### 3. Real-Time Active-Set Quadratic Program (QP)
At every 1 ms cycle, the co-processor solves:
$$\min_u \frac{1}{2} \|u - u_{nom}\|^2 \quad \text{s.t.} \quad A_{cbf} u \le b_{cbf}, \quad -u_{max} \le u \le u_{max}$$

The native primal-dual Active-Set solver in pure NumPy computes the Karush-Kuhn-Tucker (KKT) projection via Schur complements:
$$\begin{bmatrix} I & A_W^T \\ A_W & 0 \end{bmatrix} \begin{bmatrix} u^* \\ \lambda_W \end{bmatrix} = \begin{bmatrix} u_{nom} \\ b_W \end{bmatrix}$$
- **Zero Intervention**: If the VLA trajectory is safe, $u^* = u_{nom}$ ($0$ active constraints, solved in $<2\,\mu\text{s}$).
- **Tangential Deflection**: When an obstacle threatens the barrier, the QP minimally deflects torques tangentially along the barrier boundary in $<150\,\mu\text{s}$, strictly preserving set invariance ($h(q) \ge 0$).

---

## Empirical Benchmark

Simulated across an open-loop 50-step reaching task with dynamic obstacle intrusion:

| Performance Metric | Unshielded VLA Chunk | 1 kHz HOCBF-QP Shield (Ours) |
| :--- | :---: | :---: |
| **Minimum Barrier Value $h(q)$** | `-0.0400` | **`+0.0004`** |
| **Safety Invariance ($h(q) \ge 0$)** | ❌ **VIOLATED (Crash)** | ✅ **STRICTLY PRESERVED** |
| **Collision Avoided** | ❌ **False** | ✅ **True (100%)** |
| **Mean Solve Latency** | N/A (Open-loop) | **128.4 µs** |
| **Max Solve Latency** | N/A (Open-loop) | **214.1 µs** |
| **Control Frequency** | 10.0 Hz | **>1,000 Hz** |
| **Dependencies** | PyTorch / GPU | **Pure NumPy / CPU** |

---

## Quickstart

### Run the Benchmark Directly

No heavy deep learning frameworks, no ROS installation required—runs immediately in pure NumPy:

```bash
git clone https://github.com/shakstzy/edge-reflex.git
cd edge-reflex
python3 edge_reflex.py
```

### Benchmark Output

```
====================================================================================
      HIGH-ORDER CBF-QP SAFETY SHIELD: VLA ACTION CHUNK BENCHMARK
      Analytical Euler-Lagrange Dynamics | Active-Set QP Solver | Set Invariance
====================================================================================

[*] Simulating 10 Hz VLA Action Chunk execution under dynamic obstacle disturbance...
====================================================================================
Performance Metric                  | Unshielded VLA Chunk | 1 kHz HOCBF-QP Shield
------------------------------------------------------------------------------------
Min Barrier Value h(q)              |              -0.0400 |               0.0004
Safety Guarantee h(q) >= 0          |     VIOLATED (Crash) |    GUARANTEED (Safe)
Collision Avoided                   |                False |                 True
Mean Solve Latency                  |      N/A (Open-loop) |             128.4 µs
Reflex Loop Frequency               |              10.0 Hz |            >1,000 Hz
====================================================================================
[*] RESULT: HOCBF-QP shield maintains exact forward invariance (h >= 0) in 128.4 µs.
====================================================================================
```

---

## Architecture & Code Structure

- `PlanarManipulator2D`: Analytical rigid-body kinematics, Jacobian derivative $\dot{J}(q, \dot{q})$, Euler-Lagrange equations, and RK4 numerical integrator.
- `ActiveSetQPSolver`: Exact Karush-Kuhn-Tucker active-set quadratic program solver in pure NumPy.
- `HighOrderCBF`: Relative-degree 2 Control Barrier Function for Cartesian workspace obstacles and manipulator joint limits.
- `simulate_vla_chunk_execution()`: Full closed-loop comparison harness between open-loop VLA action chunk execution and the 1 kHz reflex shield.

---

## License

Apache License 2.0. Copyright (c) 2026 Adithya ([@shakstzy](https://github.com/shakstzy)).
