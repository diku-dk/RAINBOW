import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.math.vector3 as V3


def show_simple_setup():
    skeleton = IK.create_skeleton()
    B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(1), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

    IK.update_skeleton(skeleton)
    chains = IK.make_chains(skeleton)
    chains[0].goal = V3.make(0, 3, 0)

    IK.solve(chains, skeleton)
    for i in range(3):
        print(skeleton.bones[i].t_wcs)


if __name__ == '__main__':
    show_simple_setup()
