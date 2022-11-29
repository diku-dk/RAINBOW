import unittest
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.math.vector3 as V3
import rainbow.util.test_tools as TEST


class TestInverseKinematics(unittest.TestCase):

    def test1_update_skeleton(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=IK.degrees_to_radians(-90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B3 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(-90), beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0)
        IK.update_skeleton(skeleton)
        self.assertTrue(TEST.is_array_equal(B0.t_wcs, V3.make(1.0, 0.0, 0.0)))
        self.assertTrue(TEST.is_array_equal(B1.t_wcs, V3.make(1.0, 1.0, 0.0)))
        self.assertTrue(TEST.is_array_equal(B2.t_wcs, V3.make(0.0, 1.0, 0.0)))
        self.assertTrue(TEST.is_array_equal(B3.t_wcs, V3.make(1.0, -1.0, 0.0)))

    def test2_update_skeleton(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        IK.update_skeleton(skeleton)
        self.assertTrue(TEST.is_array_equal(B0.t_wcs, V3.make(1.0, 0.0, 0.0)))
        self.assertTrue(TEST.is_array_equal(B1.t_wcs, V3.make(2.0, 0.0, 0.0)))
        B0.alpha = IK.degrees_to_radians(90)
        IK.update_skeleton(skeleton)
        self.assertTrue(TEST.is_array_equal(B0.t_wcs, V3.make(1.0, 0.0, 0.0)))
        self.assertTrue(TEST.is_array_equal(B1.t_wcs, V3.make(1.0, 1.0, 0.0)))

    def test1_make_chains(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        chains = IK.make_chains(skeleton)
        self.assertTrue(len(chains) == 1)
        self.assertTrue(len(chains[0].bones) == 2)
        self.assertTrue(chains[0].bones[0] == 0)
        self.assertTrue(chains[0].bones[1] == 1)

    def test2_make_chains(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B2 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=IK.degrees_to_radians(-90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        chains = IK.make_chains(skeleton)
        self.assertTrue(len(chains) == 2)
        self.assertTrue(len(chains[0].bones) == 2)
        self.assertTrue(chains[0].bones[0] == 0)
        self.assertTrue(chains[0].bones[1] == 1)
        self.assertTrue(len(chains[1].bones) == 3)
        self.assertTrue(chains[1].bones[0] == 0)
        self.assertTrue(chains[1].bones[1] == 2)
        self.assertTrue(chains[1].bones[2] == 3)

    def test_angles(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(-90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=IK.degrees_to_radians(90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B3 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(-90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        angles = IK.get_joint_angles(skeleton)
        self.assertTrue(len(angles) == 12)
        self.assertAlmostEqual(angles[0], IK.degrees_to_radians(90), delta=0.001)
        self.assertAlmostEqual(angles[1], IK.degrees_to_radians(0), delta=0.001)
        self.assertAlmostEqual(angles[2], IK.degrees_to_radians(0), delta=0.001)
        self.assertAlmostEqual(angles[3], IK.degrees_to_radians(-90), delta=0.001)
        self.assertAlmostEqual(angles[4], IK.degrees_to_radians(0), delta=0.001)
        self.assertAlmostEqual(angles[4], IK.degrees_to_radians(0), delta=0.001)
        angles = [n for n in range(0, 12)]
        IK.set_joint_angles(skeleton, angles)
        self.assertAlmostEqual(B3.alpha, 9, delta=0.001)
        self.assertAlmostEqual(B3.beta, 10, delta=0.001)
        self.assertAlmostEqual(B3.gamma, 11, delta=0.001)

    def test_jacobian(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        IK.update_skeleton(skeleton)
        chains = IK.make_chains(skeleton)
        Jtst = IK.compute_jacobian(chains, skeleton)
        J = np.array(
            [[0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [2., 0., 2., 1., 0., 1., 0., 0., 0.],
            [0., -2., 0., 0., -1., 0., 0., 0., 0.]]
        )
        self.assertTrue(TEST.is_array_equal(J, Jtst))

    def test_hessian(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        IK.update_skeleton(skeleton)
        chains = IK.make_chains(skeleton)
        chains[0].goal = V3.make(10, 0, 0)
        J = IK.compute_jacobian(chains, skeleton)
        H = IK.compute_hessian(chains, skeleton, J)
        H_app = IK.compute_finite_difference_hessian(chains, skeleton, 0.00001)
        self.assertTrue(TEST.is_array_equal(H, H_app, dec=4))

    def test_gradient(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
        IK.update_skeleton(skeleton)
        chains = IK.make_chains(skeleton)
        chains[0].goal = V3.make(10, 0, 0)
        J = IK.compute_jacobian(chains, skeleton)
        g = IK.compute_gradient(chains, skeleton, J)
        g_app = IK.compute_finite_difference_gradient(chains, skeleton, h=0.00001)
        self.assertTrue(TEST.is_array_equal(g, g_app, dec=4))
