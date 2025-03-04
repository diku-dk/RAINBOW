import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps

from mpl_toolkits.mplot3d import Axes3D
#from stl import mesh

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.types as TYPES
import rainbow.procedural.gears as GEAR
import rainbow.simulators.prox_rigid_bodies.scenes as PROC


ps.init()

spec = GEAR.GearSpec(1, 17, bevel_cone_angle=20)
factory = GEAR.GearFactory()
gear = factory.create_gear(spec, 10)

ps_mesh = ps.register_surface_mesh("gear", gear.mesh.V, gear.mesh.T)
# show mesh grid

ps.show()