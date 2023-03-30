import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.simulators.inverse_kinematics.visualizer as GP
import rainbow.math.vector3 as V3


def show_simple_setup():
    skeleton = IK.create_skeleton()
#    B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
    B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
    B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
    B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
    IK.update_skeleton(skeleton)
    chains = IK.make_chains(skeleton)
#    chains[0].goal = V3.make(1, 1, 0)
    IK.solve(chains, skeleton)
#    print(B1)
#    print(skeleton)
    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
    S.visualize()

if __name__ == '__main__':
    show_simple_setup()
