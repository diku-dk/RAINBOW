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
    B0 = IK.create_root(skeleton, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    B4 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    B5 = IK.add_bone(skeleton, parent_idx=B4.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))

    C0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    C1 = IK.add_bone(skeleton, parent_idx=C0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    C2 = IK.add_bone(skeleton, parent_idx=C1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    
    D0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    D1 = IK.add_bone(skeleton, parent_idx=D0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    D2 = IK.add_bone(skeleton, parent_idx=D1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    
    E0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    E1 = IK.add_bone(skeleton, parent_idx=E0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    E2 = IK.add_bone(skeleton, parent_idx=E1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    
    F0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    F1 = IK.add_bone(skeleton, parent_idx=F0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
    F2 = IK.add_bone(skeleton, parent_idx=F1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))

    
#    C0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=180.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
#    C1 = IK.add_bone(skeleton, parent_idx=C0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
#    C2 = IK.add_bone(skeleton, parent_idx=C1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
#    C3 = IK.add_bone(skeleton, parent_idx=C2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
#    D0 = IK.add_bone(skeleton, parent_idx=C1.idx, alpha=180.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
#    D1 = IK.add_bone(skeleton, parent_idx=D0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
#    D2 = IK.add_bone(skeleton, parent_idx=D1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(0, 360), betaLim=(0, 360), gammaLim=(0, 360))
#    B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
#    B4 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
#    B5 = IK.add_bone(skeleton, parent_idx=B4.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
#    B6 = IK.add_bone(skeleton, parent_idx=B5.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
#    B7 = IK.add_bone(skeleton, parent_idx=B6.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
    IK.update_skeleton(skeleton)
    chains = IK.make_chains(skeleton)
    chains[0].goal = V3.make(0, 7, 0)
    chains[1].goal = V3.make(-3, 0, 0)
    chains[2].goal = V3.make(3, 0, 0)
    chains[3].goal = V3.make(-3, 5, 0)
    chains[4].goal = V3.make(3, 5, 0)
    IK.update_skeleton(skeleton)
    IK.update_skeleton(skeleton)
#    chains[2].goal = V3.make(0, 3, 0)
    IK.solve(chains, skeleton)
    IK.solve(chains, skeleton)
#    print(B1)
#    print(skeleton)
    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
    S.visualize()

if __name__ == '__main__':
    show_simple_setup()
