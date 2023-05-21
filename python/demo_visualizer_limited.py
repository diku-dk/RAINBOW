import os
import sys
import numpy as np
import cProfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.simulators.inverse_kinematics.visualizer as GP
import rainbow.math.vector3 as V3


def show_simple_setup():
    skeleton = IK.create_skeleton()
    B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 30), betaLim=(0, 30), gammaLim=(0, 0))
    B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

    IK.update_skeleton(skeleton)
    chains = IK.make_chains(skeleton)
    chains[0].goal = V3.make(0.95, 2.754141, 0)
    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
    IK.solveVariables(chains, skeleton, 300, 0.38, 0.0005, 0.0005, 0.21, 0.0001)

    S.update_mesh(skeleton, chains)

    S.visualize()

if __name__ == '__main__':
    show_simple_setup()
