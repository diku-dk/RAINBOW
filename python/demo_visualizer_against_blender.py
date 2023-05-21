import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.simulators.inverse_kinematics.visualizer as GP
import rainbow.math.vector3 as V3
from timeit import default_timer
TIMETOCREATESKELETON = 0.0
TIMETOCREATEBONES = 0.0
TIMETOUPDATESKELETON = 0.0
TIMETOMAKECHAINS = 0.0
TIMETOSOLVE = 0.0


def show_simple_setup():
    start = default_timer()
    skeleton = IK.create_skeleton()
    global TIMETOCREATESKELETON 
    TIMETOCREATESKELETON += default_timer() - start
    
    
    start = default_timer()
#    B0 = IK.create_root(skeleton, alpha=IK.degrees_to_radians(90), beta=0.0, gamma=0.0, tx=1.0, ty=0.0, tz=0.0)
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
    global TIMETOCREATEBONES
    TIMETOCREATEBONES += default_timer() - start

    start = default_timer()
    IK.update_skeleton(skeleton)
    global TIMETOUPDATESKELETON
    TIMETOUPDATESKELETON += default_timer() - start

    start = default_timer()
    chains = IK.make_chains(skeleton)
    global TIMETOMAKECHAINS
    TIMETOMAKECHAINS += default_timer() - start
    
    chains[0].goal = V3.make(0, 6, 0)
    chains[1].goal = V3.make(3.5, 2, 0)
    chains[2].goal = V3.make(-3.5, 2, 0)
    chains[3].goal = V3.make(1.5, -2.5, 0)
    chains[4].goal = V3.make(-1.5, -2.5, 0)
    print(B2.children)
#    IK.set_chain_length(chains, [7, 5, 5])
#    print(chains[0])
#    return
#    IK.update_skeleton(skeleton)

    S = GP.GraphicsComponent()
    S.generateSkeletonMesh(skeleton, chains)
   

    
    start = default_timer()
    IK.solveVariables(chains, skeleton, 1000, 0.38, 0.0001, 0.05, 0.21, 0.0001)
    global TIMETOSOLVE
    TIMETOSOLVE += default_timer() - start    
    
    S.update_mesh(skeleton, chains)
    print(B1)
    print(skeleton)
    S.visualize()

if __name__ == '__main__':
    for i in range(1):
        show_simple_setup()
    print("Time to create skeleton: " + str(TIMETOCREATESKELETON))
    print("Time to create bones: " + str(TIMETOCREATEBONES))
    print("Time to update skeleton: " + str(TIMETOUPDATESKELETON))
    print("Time to make chains: " + str(TIMETOMAKECHAINS))
    print("Time to solve: " + str(TIMETOSOLVE))

