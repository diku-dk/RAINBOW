import unittest
import os
import sys
import numpy as np
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.math.vector3 as V3
import rainbow.simulators.inverse_kinematics.visualizer as GP
import rainbow.util.test_tools as TEST


class TestInverseKinematicsSecond(unittest.TestCase):
    def compute_individual_objectives(self, chains, skeleton):
        f = []
        for chain in chains:
            e = IK.get_end_effector(chain, skeleton)
            r = chain.goal - e
            f.append(np.dot(r, r))
        for i in range(len(f)):
            f[i] *= 0.5
        return f

    def test_gradient_descent_large_skeleton(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B4 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        
        E0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        E1 = IK.add_bone(skeleton, parent_idx=E0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        E2 = IK.add_bone(skeleton, parent_idx=E1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        E3 = IK.add_bone(skeleton, parent_idx=E2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))   

        F0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        F1 = IK.add_bone(skeleton, parent_idx=F0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        F2 = IK.add_bone(skeleton, parent_idx=F1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        F3 = IK.add_bone(skeleton, parent_idx=F2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

        G0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        G1 = IK.add_bone(skeleton, parent_idx=G0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        G2 = IK.add_bone(skeleton, parent_idx=G1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        G3 = IK.add_bone(skeleton, parent_idx=G2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        
        H0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        H1 = IK.add_bone(skeleton, parent_idx=H0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        H2 = IK.add_bone(skeleton, parent_idx=H1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        H3 = IK.add_bone(skeleton, parent_idx=H2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        chains = IK.make_chains(skeleton)
        
        IK.update_skeleton(skeleton)

        chains[0].goal = V3.make(0, 3.99999997e+00, 0)
        chains[1].goal = V3.make(2.88378415, 2.17211945, 0)
        chains[2].goal = V3.make(-2.88350176, 2.17320381, 0)
        chains[3].goal = V3.make(1.47121935, -2.44213624, 0)
        chains[4].goal = V3.make(-1.47121935, -2.44213624, 0)
        
        IK.solveVariables(chains, skeleton, 1000, 0.38, 0.0001, 0.05, 0.21, 0.0001)
        errors = self.compute_individual_objectives(chains, skeleton)
        for i in range(len(errors)):
            self.assertTrue(errors[i] < 0.002)
            
    def test_gradient_descent_large_skeleton_its(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B4 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        
        E0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        E1 = IK.add_bone(skeleton, parent_idx=E0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        E2 = IK.add_bone(skeleton, parent_idx=E1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        E3 = IK.add_bone(skeleton, parent_idx=E2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))   

        F0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        F1 = IK.add_bone(skeleton, parent_idx=F0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        F2 = IK.add_bone(skeleton, parent_idx=F1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        F3 = IK.add_bone(skeleton, parent_idx=F2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

        G0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        G1 = IK.add_bone(skeleton, parent_idx=G0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        G2 = IK.add_bone(skeleton, parent_idx=G1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        G3 = IK.add_bone(skeleton, parent_idx=G2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        
        H0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        H1 = IK.add_bone(skeleton, parent_idx=H0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        H2 = IK.add_bone(skeleton, parent_idx=H1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        H3 = IK.add_bone(skeleton, parent_idx=H2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        chains = IK.make_chains(skeleton)
        
        IK.update_skeleton(skeleton)

        chains[0].goal = V3.make(0, 6, 0)
        chains[1].goal = V3.make(3.5, 2, 0)
        chains[2].goal = V3.make(-3.5, 2, 0)
        chains[3].goal = V3.make(1.5, -2.5, 0)
        chains[4].goal = V3.make(-1.5, -2.5, 0)
        
        its = IK.solveVariables(chains, skeleton, 1000, 0.38, 0.0001, 0.05, 0.21, 0.0001)
        self.assertTrue(len(its) == 87)
        
    def test_limits_static(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 30), betaLim=(0, 30), gammaLim=(0, 0))
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

        IK.update_skeleton(skeleton)
        chains = IK.make_chains(skeleton)
        chains[0].goal = V3.make(0.95, 2.754141, 0)
        IK.solveVariables(chains, skeleton, 300, 0.38, 0.0001, 0.05, 0.21, 0.0001)
        bone = skeleton.bones[0]
        self.assertTrue(bone.alphaMin > -0.0001)
        self.assertTrue(bone.alphaMax < 30.0001)
        self.assertTrue(bone.betaMin > -0.0001)
        self.assertTrue(bone.betaMax < 30.0001)
        self.assertTrue(bone.gammaMin > -0.0001)
        self.assertTrue(bone.gammaMax < 0.0001)

    def test_limits_random(self):
        for i in range(50):
            skeleton = IK.create_skeleton()
            B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 30), betaLim=(0, 30), gammaLim=(0, 0))
            B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
            B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
            B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

            IK.update_skeleton(skeleton)
            chains = IK.make_chains(skeleton)
            chains[0].goal = V3.make(random.uniform(-3.05, 3.05), random.uniform(-3.05, 3.05), random.uniform(-3.05, 3.05))
            IK.solveVariables(chains, skeleton, 100, 0.38, 0.0001, 0.05, 0.21, 0.0001)
            bone = skeleton.bones[0]
            self.assertTrue(bone.alphaMin > -0.0001)
            self.assertTrue(bone.alphaMax < 30.0001)
            self.assertTrue(bone.betaMin > -0.0001)
            self.assertTrue(bone.betaMax < 30.0001)
            self.assertTrue(bone.gammaMin > -0.0001)
            self.assertTrue(bone.gammaMax < 0.0001)

    def test_limits_no_difference(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

        IK.update_skeleton(skeleton)
        chains = IK.make_chains(skeleton)
        chains[0].goal = V3.make(0.95, 2.754141, 0)
        firstIts = len(IK.solveVariables(chains, skeleton, 100, 0.38, 0.0001, 0.05, 0.21, 0.0001))

        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-5120, 5120), betaLim=(-5120, 5120), gammaLim=(-5120, 5120))
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-5120, 5120), betaLim=(-5120, 5120), gammaLim=(-5120, 5120))
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-5120, 5120), betaLim=(-5120, 5120), gammaLim=(-5120, 5120))
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-5120, 5120), betaLim=(-5120, 5120), gammaLim=(-5120, 5120))

        IK.update_skeleton(skeleton)
        chains = IK.make_chains(skeleton)
        chains[0].goal = V3.make(0.95, 2.754141, 0)
        secondIts = len(IK.solveVariables(chains, skeleton, 100, 0.38, 0.0001, 0.05, 0.21, 0.0001))
        self.assertTrue(firstIts, secondIts)
            

    def test_gradient_descent(self):
        skeleton = IK.create_skeleton()
        B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
        B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

        IK.update_skeleton(skeleton)
        chains = IK.make_chains(skeleton)
        chains[0].goal = V3.make(0.95, 2.754141, 0)
        IK.solveVariables(chains, skeleton, 200, 0.38, 0.0001, 0.05, 0.21, 0.0001)
        self.assertTrue(IK.compute_objective(chains, skeleton) < 0.005)

    def test_gradient_descent_random(self):
        #Make sure it can reach the goal. If the solver gets limited by iterations
        #ignore that test, as it can't be used.
        for i in range(50):
            skeleton = IK.create_skeleton()
            B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
            B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
            B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
            B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

            IK.update_skeleton(skeleton)
            chains = IK.make_chains(skeleton)
            chains[0].goal = V3.make(random.uniform(-2.95, 2.95), random.uniform(-2.95, 2.95), random.uniform(-2.95, 2.95))
            if len(IK.solveVariables(chains, skeleton, 200, 0.38, 0.0001, 0.05, 0.21, 0.0001)) > 200:
                self.assertTrue(IK.compute_objective(chains, skeleton) < 0.05)

            
if __name__ == '__main__':
    unittest.main()
