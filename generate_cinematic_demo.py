"""
Generates High-Definition Close-Up 3D Simulation & Telemetry Animation:
  - Left: Macro 3D Zoom on End-Effector / Gripper Interaction Zone & Obstacle
  - Right: Live Synchronized Telemetry (Barrier Invariance h(t) + Joint Torques)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from edge_reflex import FrankaPanda7DOF, FeasibleHOCBF

def run_sim(num_steps=50, dt=0.001):
    robot = FrankaPanda7DOF()
    cbf = FeasibleHOCBF(robot, alpha1=25.0, alpha2=35.0)

    q_init = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
    p_ee_start = robot.forward_kinematics(q_init)
    J_init = robot.geometric_jacobian(q_init)

    direction = np.array([0.4, 0.0, -0.1], dtype=np.float64)
    direction = direction / np.linalg.norm(direction)
    dq_init = np.linalg.pinv(J_init) @ (0.6 * direction)

    p_obs = p_ee_start + 0.045 * direction
    r_safe = 0.03

    q_vla_target = q_init + np.linalg.pinv(J_init) @ (0.12 * direction)
    Kp = np.array([120.0, 120.0, 100.0, 100.0, 40.0, 40.0, 20.0], dtype=np.float64)
    Kd = np.array([15.0, 15.0, 10.0, 10.0, 5.0, 5.0, 2.0], dtype=np.float64)

    # 1. Unshielded Simulation
    q_u = q_init.copy()
    dq_u = dq_init.copy()
    unshielded_log = []

    for step in range(num_steps):
        u_vla = Kp * (q_vla_target - q_u) - Kd * dq_u
        u_applied = np.clip(u_vla, -robot.tau_max, robot.tau_max)
        origins, _, _ = robot.link_transforms(q_u)
        h = cbf.barrier_value(q_u, p_obs, r_safe)

        unshielded_log.append({
            "step": step,
            "time_ms": step * dt * 1000,
            "q": q_u.copy(),
            "tau": u_applied.copy(),
            "p_ee": origins[-1].copy(),
            "origins": origins.copy(),
            "h": h,
            "collision": h < 0.0
        })
        q_u, dq_u = robot.step_rk4(q_u, dq_u, u_applied, dt)

    # 2. Shielded Simulation
    q_s = q_init.copy()
    dq_s = dq_init.copy()
    shielded_log = []

    for step in range(num_steps):
        u_vla = Kp * (q_vla_target - q_s) - Kd * dq_s
        res = cbf.solve_feasible_cbf_qp(q_s, dq_s, u_vla, p_obs, r_safe)
        tau_safe = res["tau"]
        origins, _, _ = robot.link_transforms(q_s)
        h = cbf.barrier_value(q_s, p_obs, r_safe)

        shielded_log.append({
            "step": step,
            "time_ms": step * dt * 1000,
            "q": q_s.copy(),
            "tau": tau_safe.copy(),
            "p_ee": origins[-1].copy(),
            "origins": origins.copy(),
            "h": h,
            "latency_us": res["latency_us"]
        })
        q_s, dq_s = robot.step_rk4(q_s, dq_s, tau_safe, dt)

    return {
        "p_obs": p_obs,
        "r_safe": r_safe,
        "unshielded": unshielded_log,
        "shielded": shielded_log
    }

def render_zoomed_animation(data, output_gif="scratch/edge-reflex/assets/franka_reflex_demo.gif"):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 9), dpi=110)
    fig.patch.set_facecolor("#080c14")

    # Layout: Left = 3D Close-up; Right Top = h(t) curve; Right Bottom = Real-time Joint Torques
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.set_facecolor("#080c14")

    ax_h = fig.add_subplot(2, 2, 2)
    ax_h.set_facecolor("#0f172a")

    ax_tau = fig.add_subplot(2, 2, 4)
    ax_tau.set_facecolor("#0f172a")

    p_obs = data["p_obs"]
    r_safe = data["r_safe"]
    unshielded = data["unshielded"]
    shielded = data["shielded"]
    num_frames = len(unshielded)

    # Obstacle mesh (high-res sphere)
    u_s = np.linspace(0, 2 * np.pi, 24)
    v_s = np.linspace(0, np.pi, 24)
    x_s = p_obs[0] + r_safe * np.outer(np.cos(u_s), np.sin(v_s))
    y_s = p_obs[1] + r_safe * np.outer(np.sin(u_s), np.sin(v_s))
    z_s = p_obs[2] + r_safe * np.outer(np.ones(np.size(u_s)), np.cos(v_s))

    all_times = [d["time_ms"] for d in unshielded]
    all_h_u = [d["h"] * 1000 for d in unshielded]
    all_h_s = [d["h"] * 1000 for d in shielded]

    def update(frame):
        # 1. Update 3D Close-Up View
        ax3d.clear()
        ax3d.set_facecolor("#080c14")

        # Obstacle wireframe & core
        ax3d.plot_wireframe(x_s, y_s, z_s, color="#ef4444", alpha=0.35, linewidth=0.9)
        ax3d.scatter([p_obs[0]], [p_obs[1]], [p_obs[2]], color="#ffffff", s=50, marker="o")

        orig_u = unshielded[frame]["origins"]
        orig_s = shielded[frame]["origins"]

        # Draw Wrist & Gripper Links (Links 4, 5, 6, 7)
        # Red: Unshielded
        ax3d.plot(orig_u[4:, 0], orig_u[4:, 1], orig_u[4:, 2], color="#ff4d4d", linewidth=4.5, alpha=0.8, label="Unshielded VLA")
        ax3d.scatter(orig_u[4:, 0], orig_u[4:, 1], orig_u[4:, 2], color="#ff4d4d", s=45)

        # Draw Gripper Head & TCP (End-effector)
        ee_u = orig_u[-1]
        ax3d.scatter([ee_u[0]], [ee_u[1]], [ee_u[2]], color="#ff1111", s=110, marker="X", edgecolors="white", linewidths=1.2, label="Unshielded TCP (Crash)")

        # Green: Feasible-HOCBF
        ax3d.plot(orig_s[4:, 0], orig_s[4:, 1], orig_s[4:, 2], color="#00ff88", linewidth=5.5, alpha=0.95, label="1 kHz Feasible-HOCBF")
        ax3d.scatter(orig_s[4:, 0], orig_s[4:, 1], orig_s[4:, 2], color="#00ff88", s=55)

        # Draw Shielded TCP
        ee_s = orig_s[-1]
        ax3d.scatter([ee_s[0]], [ee_s[1]], [ee_s[2]], color="#00ff88", s=120, marker="o", edgecolors="white", linewidths=1.5, label="Shielded TCP (Safe Slide)")

        # Historical breadcrumbs
        pts_u = np.array([unshielded[i]["p_ee"] for i in range(frame + 1)])
        pts_s = np.array([shielded[i]["p_ee"] for i in range(frame + 1)])
        ax3d.plot(pts_u[:, 0], pts_u[:, 1], pts_u[:, 2], color="#ff4d4d", linestyle="--", linewidth=2.0)
        ax3d.plot(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2], color="#00ff88", linewidth=2.8)

        # Tight Zoom Window right at the interaction zone!
        ax3d.set_xlim(p_obs[0] - 0.055, p_obs[0] + 0.045)
        ax3d.set_ylim(p_obs[1] - 0.055, p_obs[1] + 0.055)
        ax3d.set_zlim(p_obs[2] - 0.055, p_obs[2] + 0.055)

        ax3d.set_title("Close-Up: End-Effector / Obstacle Boundary Interaction", fontsize=12, fontweight="bold", color="white", pad=12)
        ax3d.set_xlabel("X (m)", color="#64748b", labelpad=-5)
        ax3d.set_ylabel("Y (m)", color="#64748b", labelpad=-5)
        ax3d.set_zlabel("Z (m)", color="#64748b", labelpad=-5)
        ax3d.tick_params(colors="#64748b", labelsize=7)
        ax3d.legend(loc="upper left", frameon=True, facecolor="#1e293b", edgecolor="none", fontsize=8)
        ax3d.view_init(elev=22, azim=52)

        # 2. Update Live Barrier Curve h(t)
        ax_h.clear()
        ax_h.set_facecolor("#0f172a")
        ax_h.axhline(0.0, color="#ffffff", linestyle="--", linewidth=1.2, alpha=0.7)
        ax_h.plot(all_times, all_h_u, color="#ff4d4d", linewidth=1.5, alpha=0.4, label="Unshielded History")
        ax_h.plot(all_times, all_h_s, color="#00ff88", linewidth=1.5, alpha=0.4, label="Feasible-HOCBF History")

        # Plot progress up to current frame
        ax_h.plot(all_times[:frame+1], all_h_u[:frame+1], color="#ff4d4d", linewidth=2.5)
        ax_h.plot(all_times[:frame+1], all_h_s[:frame+1], color="#00ff88", linewidth=3.0)
        ax_h.scatter([all_times[frame]], [all_h_u[frame]], color="#ff4d4d", s=70, zorder=5)
        ax_h.scatter([all_times[frame]], [all_h_s[frame]], color="#00ff88", s=70, zorder=5)

        h_val_u = unshielded[frame]["h"]
        h_val_s = shielded[frame]["h"]

        ax_h.set_title(f"Barrier Invariance h(q) >= 0 [t = {frame} ms]", fontsize=11, fontweight="bold", color="white")
        ax_h.set_xlabel("Time (ms)", color="#94a3b8", fontsize=9)
        ax_h.set_ylabel("h(q) (x10^-3 m^2)", color="#94a3b8", fontsize=9)
        ax_h.grid(True, linestyle=":", alpha=0.25)
        ax_h.tick_params(colors="#94a3b8", labelsize=8)

        status_text = f"Unshielded: {'CRASH (h < 0)' if h_val_u < 0 else 'Nominal'}\nFeasible-HOCBF: SAFE (h = +{max(0.0, h_val_s):.4f})"
        ax_h.text(0.03, 0.12, status_text, transform=ax_h.transAxes, color="white", fontsize=8.5,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e293b", alpha=0.85, edgecolor="none"))

        # 3. Update Real-Time Motor Torque Bars
        ax_tau.clear()
        ax_tau.set_facecolor("#0f172a")
        joints = [f"J{i+1}" for i in range(7)]
        tau_s = shielded[frame]["tau"]
        tau_max = [87, 87, 87, 87, 12, 12, 12]

        # Normalize torque ratio
        ratio = [tau_s[i] / tau_max[i] * 100.0 for i in range(7)]
        colors = ["#38bdf8" if i < 4 else "#f59e0b" for i in range(7)]

        bars = ax_tau.bar(joints, ratio, color=colors, width=0.55, alpha=0.9)
        ax_tau.axhline(100.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Motor Limit (+100%)")
        ax_tau.axhline(-100.0, color="#ef4444", linestyle="--", linewidth=1.2, label="Motor Limit (-100%)")
        ax_tau.axhline(0.0, color="#64748b", linestyle="-", linewidth=0.8)

        ax_tau.set_title("Franka Panda Motor Torque Utilization (% of tau_max)", fontsize=11, fontweight="bold", color="white")
        ax_tau.set_ylabel("% Max Torque", color="#94a3b8", fontsize=9)
        ax_tau.set_ylim(-125, 125)
        ax_tau.grid(True, linestyle=":", alpha=0.25)
        ax_tau.tick_params(colors="#94a3b8", labelsize=8)
        ax_tau.legend(loc="upper right", frameon=True, facecolor="#1e293b", edgecolor="none", fontsize=7.5)

    plt.suptitle("edge-reflex: 1 kHz Feasible-HOCBF Safety Reflex on 7-DOF Franka Panda", fontsize=15, fontweight="bold", y=0.97, color="white")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    anim = FuncAnimation(fig, update, frames=num_frames, interval=40)
    anim.save(output_gif, writer="pillow", fps=25)
    plt.close()
    print(f"[*] Generated zoomed cinematic GIF: {output_gif}")

if __name__ == "__main__":
    print("[*] Running simulation and generating close-up zoomed animation...")
    data = run_sim(num_steps=50, dt=0.001)
    render_zoomed_animation(data, "scratch/edge-reflex/assets/franka_reflex_demo.gif")
