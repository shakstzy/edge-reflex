"""
Unit tests for Feasible-HOCBF 1 kHz Reflex Shield for 7-DOF Franka Emika Panda.
"""

import unittest
import numpy as np
from edge_reflex import FrankaPanda7DOF, FeasibleHOCBF, simulate_franka_vla_benchmark


class TestEdgeReflex(unittest.TestCase):

    def setUp(self):
        self.robot = FrankaPanda7DOF()
        self.cbf = FeasibleHOCBF(self.robot, alpha1=40.0, alpha2=60.0)

    def test_7dof_kinematics_and_dynamics(self):
        q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        p_ee = self.robot.forward_kinematics(q)
        self.assertEqual(p_ee.shape, (3,))

        J = self.robot.geometric_jacobian(q)
        self.assertEqual(J.shape, (3, 7))

        M = self.robot.mass_matrix(q)
        self.assertEqual(M.shape, (7, 7))
        eigvals = np.linalg.eigvalsh(M)
        self.assertTrue(np.all(eigvals > 0))

    def test_torque_saturation_feasibility_governor(self):
        q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
        p_ee = self.robot.forward_kinematics(q)
        J = self.robot.geometric_jacobian(q)

        direction = np.array([0.0, 1.0, 0.0])
        v_des = 1.8 * direction
        dq = np.linalg.pinv(J) @ v_des

        p_obs = p_ee + 0.025 * direction
        r_safe = 0.02
        u_nom = np.zeros(7)

        # Naive fails due to wrist torque saturation
        res_naive = self.cbf.solve_naive_cbf_qp(q, dq, u_nom, p_obs, r_safe)
        self.assertFalse(res_naive["feasible"])

        # Feasible-HOCBF guarantees primal feasibility
        res_feasible = self.cbf.solve_feasible_cbf_qp(q, dq, u_nom, p_obs, r_safe)
        self.assertTrue(res_feasible["feasible"])
        self.assertTrue(np.all(np.abs(res_feasible["tau"]) <= self.robot.tau_max + 1e-4))

    def test_1khz_benchmark(self):
        res = simulate_franka_vla_benchmark(num_steps=50, dt=0.001)
        self.assertFalse(res["unshielded"]["collision_avoided"])
        self.assertTrue(res["feasible_hocbf"]["collision_avoided"])
        self.assertLess(res["feasible_hocbf"]["avg_solve_latency_us"], 500.0)


if __name__ == "__main__":
    unittest.main()
