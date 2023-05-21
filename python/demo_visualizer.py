import os
import sys
import numpy as np
import cProfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.simulators.inverse_kinematics.visualizer as GP
import rainbow.math.vector3 as V3


def show_simple_setup(factorX, factorY):
    skeleton = IK.create_skeleton()
#    B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
    B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 30), betaLim=(0, 30), gammaLim=(0, 0))
#    B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

    IK.update_skeleton(skeleton)
    chains = IK.make_chains(skeleton)
#    IK.solve(chains, skeleton, 1000, 1, 0.0001, 0.0001, 0.1, 0.003)
#    chains[0].goal = V3.make(0.95, 0.754141, 0)
    chains[0].goal = V3.make(0.95, 2.754141, 0)
#    IK.solveVariables(chains, skeleton, 50, 1.0, 0.0005, 0.0005, 0.01+0.05*factorX, 0.0001*factorY)
#    IK.solveVariables(chains, skeleton, 50, 1.0, 0.0005, 0.0005, 0.05+0.01*factorX, 0.0001*factorY)
    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
    IK.solveVariables(chains, skeleton, 300, 0.38, 0.0005, 0.0005, 0.21, 0.0001)

    S.update_mesh(skeleton, chains)
    IK.solveVariables(chains, skeleton, 300, 0.38, 0.0005, 0.0005, 0.21, 0.0001)
#    IK.solve(chains, skeleton, 10, 1, 0.0001, 0.0001, 0.05+0.05*factorX, 0.0001*10*factorY)
#    print(B1)
#    print(skeleton)
    S.visualize()

if __name__ == '__main__':
    for i in range(1):
        for j in range(1):
            show_simple_setup(i, j+1)
