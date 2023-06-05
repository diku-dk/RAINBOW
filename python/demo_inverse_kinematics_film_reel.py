import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.simulators.inverse_kinematics.visualizer as GP
import rainbow.math.vector3 as V3

def show_simple_setup():
    skeleton = IK.create_skeleton()
    
    B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B1 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B2 = IK.add_bone(skeleton, parent_idx=B1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B3 = IK.add_bone(skeleton, parent_idx=B2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    B4 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    
    E0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    E1 = IK.add_bone(skeleton, parent_idx=E0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.75, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    E2 = IK.add_bone(skeleton, parent_idx=E1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    E3 = IK.add_bone(skeleton, parent_idx=E2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))   

    F0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    F1 = IK.add_bone(skeleton, parent_idx=F0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-0.75, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    F2 = IK.add_bone(skeleton, parent_idx=F1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    F3 = IK.add_bone(skeleton, parent_idx=F2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    
    """    E0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    E1 = IK.add_bone(skeleton, parent_idx=E0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    E2 = IK.add_bone(skeleton, parent_idx=E1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))   

    F0 = IK.add_bone(skeleton, parent_idx=B3.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    F1 = IK.add_bone(skeleton, parent_idx=F0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    F2 = IK.add_bone(skeleton, parent_idx=F1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))"""

    """    G0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    G1 = IK.add_bone(skeleton, parent_idx=G0.idx, alpha=IK.degrees_to_radians(0), beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    G2 = IK.add_bone(skeleton, parent_idx=G1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    
    H0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    H1 = IK.add_bone(skeleton, parent_idx=H0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    H2 = IK.add_bone(skeleton, parent_idx=H1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))"""

    G0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=IK.degrees_to_radians(0.0), beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    G1 = IK.add_bone(skeleton, parent_idx=G0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    G2 = IK.add_bone(skeleton, parent_idx=G1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    G3 = IK.add_bone(skeleton, parent_idx=G2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    
    H0 = IK.add_bone(skeleton, parent_idx=B0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    H1 = IK.add_bone(skeleton, parent_idx=H0.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=-1.0, ty=0.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    H2 = IK.add_bone(skeleton, parent_idx=H1.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))
    H3 = IK.add_bone(skeleton, parent_idx=H2.idx, alpha=0.0, beta=0.0, gamma=0.0, tx=0.0, ty=-1.0, tz=0.0, alphaLim=(-180, 180), betaLim=(-180, 180), gammaLim=(-180, 180))

    IK.update_skeleton(skeleton)
    chains = IK.make_chains(skeleton)
    chains[0].goal = V3.make(0, 6, 0)
    chains[1].goal = V3.make(1, 0.75, 0)
    chains[2].goal = V3.make(-1, 0.75, 0)
    chains[3].goal = V3.make(1, -3, 0)
    chains[4].goal = V3.make(-1, -3, 0)

    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
   

    
    IK.solveVariables(chains, skeleton, 1000, 0.38, 0.0001, 0.05, 0.21, 0.0001)
    
    S.update_mesh(skeleton, chains)
    S.visualize()
    
    #Frame 2:
    chains = IK.make_chains(skeleton)
    chains[0].goal = V3.make(0, 6, 0)
    chains[1].goal = V3.make(2.25, 0.5, 0)
    chains[2].goal = V3.make(-2.25, 0.5, 0)
    chains[3].goal = V3.make(1.5, -3, 0)
    chains[4].goal = V3.make(-1.5, -3, 0)

    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
   

    
    IK.solveVariables(chains, skeleton, 1000, 0.38, 0.0001, 0.05, 0.21, 0.0001)
    
    S.update_mesh(skeleton, chains)
    S.visualize()
    
    #Frame 3:
    chains = IK.make_chains(skeleton)
    chains[0].goal = V3.make(0, 6, 0)
    chains[1].goal = V3.make(3.5, 2, 0)
    chains[2].goal = V3.make(-3.5, 2, 0)
    chains[3].goal = V3.make(2.25, -3, 0)
    chains[4].goal = V3.make(-2.25, -3, 0)

    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
   

    
    IK.solveVariables(chains, skeleton, 1000, 0.38, 0.0001, 0.05, 0.21, 0.0001)
    
    S.update_mesh(skeleton, chains)
    S.visualize()
    
    #Frame 4:
    chains = IK.make_chains(skeleton)
    chains[0].goal = V3.make(0, 6, 0)
    chains[1].goal = V3.make(3.5, 4.5, 0)
    chains[2].goal = V3.make(-3.5, 4.5, 0)
    chains[3].goal = V3.make(3, -3, 0)
    chains[4].goal = V3.make(-3, -3, 0)

    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
   

    
    IK.solveVariables(chains, skeleton, 1000, 0.38, 0.0001, 0.05, 0.21, 0.0001)
    
    S.update_mesh(skeleton, chains)
    S.visualize()

if __name__ == '__main__':
    for i in range(1):
        show_simple_setup()

