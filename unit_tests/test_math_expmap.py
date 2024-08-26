import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.expmap as EXP
import rainbow.math.quaternion as Q
import rainbow.math.vector3 as V3
import rainbow.util.test_tools as TEST


class TestExomapAPI(unittest.TestCase):
    def test_log_exp_1(self):
        expected = Q.rand()
        v = EXP.log(expected)
        actual = EXP.exp(v)
        self.assertTrue(TEST.is_array_equal(abs(actual), abs(expected)))

    def test_log_exp_2(self):
        expected = Q.rand()
        v = EXP.to_expmap(expected)
        actual = EXP.to_quaternion(v)
        self.assertTrue(TEST.is_array_equal(abs(actual), abs(expected)))

    def test_log_exp_3(self):
        zeros = Q.make(0, 1, 2, 3)
        expected = V3.make(0, 0, 0)
        actual = EXP.log(zeros)
        self.assertTrue(TEST.is_array_equal(abs(actual), abs(expected)))

    def test_exp_1(self):
        const = np.pi * (1 / (np.sqrt(3)))
        v = V3.make(const, const, const)
        theta = V3.norm(v)
        sin_exp = np.sin(0.5 * theta) / theta
        expected = Q.make(
            np.cos(0.5 * theta), sin_exp * v[0], sin_exp * v[1], sin_exp * v[2]
        )
        actual = EXP.exp(v)
        self.assertTrue(TEST.is_array_equal(abs(actual), abs(expected)))

    def test_reparameterization_1(self):
        const = np.pi + np.pi / 2 + np.pi / 4
        rotation = const * (1 / np.sqrt(3))
        v = V3.make(rotation, rotation, rotation)
        expected = const
        actual = 2 * np.pi - V3.norm(EXP.reparameterization(v))
        self.assertEqual(actual, expected)

    def test_reparameterization_2(self):
        const = np.pi / 2 + np.pi
        rotation = const * (1 / np.sqrt(3))
        v = V3.make(rotation, rotation, rotation)
        expected = const
        actual = 2 * np.pi - V3.norm(EXP.reparameterization(v))
        self.assertAlmostEqual(actual, expected, 4)

    def test_reparameterization_3(self):
        const = 7 * np.pi
        rotation = const * (1 / np.sqrt(3))
        v = V3.make(rotation, rotation, rotation)
        expected = np.pi
        actual = 2 * np.pi - V3.norm(EXP.reparameterization(v))
        self.assertAlmostEqual(actual, expected, 4)

    def test_dvdt_1(self):
        const1 = np.pi + np.pi / 4
        const2 = np.pi + np.pi / 2
        quad_const = 1 / np.sqrt(3)
        rot_1 = const1 * quad_const
        rot_2 = const2 * quad_const
        omega = V3.make(rot_1, rot_1, rot_1)
        v = V3.make(rot_2, rot_2, rot_2)
        dt = 1e-7
        q = EXP.exp(v)
        angual_dt = V3.norm(omega) * dt
        unit_omega = V3.unit(omega)
        vq = Q.Ru(angual_dt, unit_omega)
        q_dt = Q.prod(vq, q)
        expected = (EXP.log(q_dt) - EXP.log(q)) / dt
        actual = EXP.dvdt(omega, v)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)

    def test_dvdt_2(self):
        const1 = np.pi + np.pi / 4
        const2 = 1e-7
        quad_const = 1 / np.sqrt(3)
        rot_1 = const1 * quad_const
        rot_2 = const2 * quad_const
        omega = V3.make(rot_1, rot_1, rot_1)
        v = V3.make(rot_2, rot_2, rot_2)
        dt = 1e-7
        q = EXP.exp(v)
        angual_dt = V3.norm(omega) * dt
        unit_omega = V3.unit(omega)
        vq = Q.Ru(angual_dt, unit_omega)
        q_dt = Q.prod(vq, q)
        expected = (EXP.log(q_dt) - EXP.log(q)) / dt
        actual = EXP.dvdt(omega, v)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)

    def test_dQdv_1(self):
        radians = np.pi / 4
        quad_const = 1 / np.sqrt(3)
        rot = radians * quad_const
        v = V3.make(rot, rot, rot)
        dt = 1e-7
        e_i = V3.make(1, 0, 0)
        dt_i = dt * e_i
        q = EXP.exp(v)
        q_dt = EXP.exp(v + dt_i)
        grad = (q_dt - q) / dt
        expected = grad
        actual = EXP.dQdv_i(v, 0)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)

    def test_dQdv_2(self):
        radians = 1e-7
        quad_const = 1 / np.sqrt(3)
        rot = radians * quad_const
        v = V3.make(rot, rot, rot)
        dt = 1e-7
        e_i = V3.make(1, 0, 0)
        dt_i = dt * e_i
        q = EXP.exp(v)
        q_dt = EXP.exp(v + dt_i)
        grad = (q_dt - q) / dt
        expected = grad
        actual = EXP.dQdv_i(v, 0)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)

    def test_dRdv_1(self):
        const1 = np.pi + np.pi / 4
        quad_const = 1 / np.sqrt(3)
        rot = const1 * quad_const
        v = V3.make(rot, rot, rot)
        v = EXP.reparameterization(v)
        dt = 1e-7
        e_i = V3.make(1, 0, 0)
        dt_i = dt * e_i
        m = EXP.to_matrix(v)
        m_dt = EXP.to_matrix(v + dt_i)
        grad = (m_dt - m) / dt
        expected = grad
        actual = EXP.dRdv_i(v, 0)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)

    def test_dRdv_2(self):
        const1 = np.pi + np.pi / 4
        quad_const = 1 / np.sqrt(3)
        rot = const1 * quad_const
        v = V3.make(rot, rot, rot)
        v = EXP.reparameterization(v)
        dt = 1e-7
        e_i = V3.make(0, 1, 0)
        dt_i = dt * e_i
        m = EXP.to_matrix(v)
        m_dt = EXP.to_matrix(v + dt_i)
        grad = (m_dt - m) / dt
        expected = grad
        actual = EXP.dRdv_i(v, 1)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)

    def test_dRdv_3(self):
        const1 = np.pi / 4
        quad_const = 1 / np.sqrt(3)
        rot = const1 * quad_const
        v = V3.make(rot, rot, rot)
        dt = 1e-7
        e_i = V3.make(0, 0, 1)
        dt_i = dt * e_i
        m = EXP.to_matrix(v)
        m_dt = EXP.to_matrix(v + dt_i)
        grad = (m_dt - m) / dt
        expected = grad
        actual = EXP.dRdv_i(v, 2)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)

    def test_dRdv_4(self):
        const1 = 1e-7
        quad_const = 1 / np.sqrt(3)
        rot = const1 * quad_const
        v = V3.make(rot, rot, rot)
        dt = 1e-7
        e_i = V3.make(1, 0, 0)
        dt_i = dt * e_i
        m = EXP.to_matrix(v)
        m_dt = EXP.to_matrix(v + dt_i)
        grad = (m_dt - m) / dt
        expected = grad
        actual = EXP.dRdv_i(v, 0)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)

    def test_dRdv_4(self):
        const1 = np.pi + np.pi / 4
        quad_const = 1 / np.sqrt(3)
        rot = const1 * quad_const
        v = V3.make(rot, rot, rot)
        v = EXP.reparameterization(v)
        dt = 1e-7
        e_i = V3.make(1, 0, 0)
        dt_i = dt * e_i
        m = EXP.to_matrix(v)
        m_dt = EXP.to_matrix(v + dt_i)
        grad = (m_dt - m) / dt
        expected = grad
        actual = EXP.dRdv_i(v, 0)
        tau = 2
        self.assertTrue(np.sum(abs(actual - expected)) <= tau * dt)
