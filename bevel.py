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

m = 1
z_sun = 31
z_planet = 17
z_ring = z_sun + 2 * z_planet
bevel_spec = GEAR.PlanetaryGearSpec(m, z_sun, z_planet, z_ring)
factory = GEAR.GearFactory()
gear1 = factory.create_planetary_gear(bevel_spec, 10)

color = V3.make( 140, 0, 0) / 255
#a4001d
""" ps.register_surface_mesh("sun_gear", gear.sun_gear.mesh.V, gear.sun_gear.mesh.T, smooth_shade=False, color=color)
for i, planet in enumerate(gear.planet_gears):
    ps_mesh = ps.register_surface_mesh(f"planet_{i}", planet.mesh.V, planet.mesh.T, smooth_shade=False, color=color)
    transformation_matrix = np.eye(4, dtype=np.float64)
    transformation_matrix[:3, :3] = Q.to_matrix(planet.orientation)
    transformation_matrix[:3, 3] = planet.position
    ps_mesh.set_transform(transformation_matrix)
ps.register_surface_mesh("ring_gear", gear.ring_gear.mesh.V, gear.ring_gear.mesh.T, smooth_shade=False, color=color) """

shaft_angle = 70
shaft_angle_rad = np.deg2rad(shaft_angle)

bevel_spec = GEAR.BevelSpec(1, 31, 51, shaft_angle)
factory = GEAR.GearFactory()
bevel_gear_pair = factory.create_bevel_gear_pair(bevel_spec, 1 / 3)
gear1 = bevel_gear_pair.gear1
gear2 = bevel_gear_pair.gear2

#gear1 = factory.create_gear(bevel_spec.spec1, bevel_spec.cone_distance / 3)
#gear2 = factory.create_gear(bevel_spec.spec2, bevel_spec.cone_distance / 3 * 31 / 51)

""" cone_height1 = bevel_spec.spec1.rp / np.tan(bevel_spec.spec1.bevel_cone_angle)
cone_height2 = bevel_spec.spec2.rp / np.tan(bevel_spec.spec2.bevel_cone_angle)

gear1_position = V3.make(0, 0, cone_height1)
gear1_orientation = Q.identity()
gear2_orientation = Q.Ry(shaft_angle_rad)
gear2_position = Q.rotate(gear2_orientation, V3.make(0, 0, cone_height2)) """

ps_mesh1 = ps.register_surface_mesh("gear1", gear1.mesh.V, gear1.mesh.T)
transformation_matrix = np.eye(4, dtype=np.float64)
transformation_matrix[:3, :3] = Q.to_matrix(gear1.orientation)
transformation_matrix[:3, 3] = gear1.position
ps_mesh1.set_transform(transformation_matrix)

ps_mesh2 = ps.register_surface_mesh("gear2", gear2.mesh.V, gear2.mesh.T)
transformation_matrix = np.eye(4, dtype=np.float64)
transformation_matrix[:3, :3] = Q.to_matrix(gear2.orientation)
transformation_matrix[:3, 3] = gear2.position
ps_mesh2.set_transform(transformation_matrix)

ps.set_ground_plane_mode('none')
ps.show()
