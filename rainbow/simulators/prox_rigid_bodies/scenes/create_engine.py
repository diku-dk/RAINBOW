import logging
import numpy as np

import rainbow.math.involute as INV
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.geometry.surface_mesh as MESH
from rainbow.simulators.prox_rigid_bodies.types import Engine

import rainbow.procedural.gears.generators as GEAR
import rainbow.procedural.gears.mating as MATING

def create_engine(
    engine: Engine,
    density: float = 1.0,
    material_name: str = "default",
    ):
    def add_gear(gear: GEAR.Gear, name: str):
        return API.add_object(
            engine, 
            name, 
            gear.mesh, 
            gear.position, 
            gear.orientation, 
            "fixed" if gear.spec.is_internal else "free",
            material_name,
            density,
        )
    
    stroke_height = 10
    
    z_ring = 60
    z_planet = int(z_ring * 2 / 3)
    m = stroke_height / (z_ring - z_planet)
    gear_width = 4 * m
    
    piston_radius = 4
    piston_height = 4
    clearance = 1
    eccentricity = 1
    disk_radius = 5
    disk_width = gear_width
    con_rod_width = disk_width * 0.8
    
    # Create the gears
    ring_spec = GEAR.GearSpec(m, z_ring, is_internal=True)
    planet_spec = GEAR.GearSpec(m, z_planet, is_internal=False)
    
    gear_factory = GEAR.GearFactory()
    
    # Add gears
    ring1 = gear_factory.create_gear(ring_spec, gear_width)
    #ring2 = gear_factory.create_gear(ring_spec, gear_width)
    planet1 = gear_factory.create_gear(planet_spec, gear_width)
    #planet2 = gear_factory.create_gear(planet_spec, gear_width)
    
    # Set the position and orientation of the first pair
    ring1.orientation = Q.Rz(-np.pi / 2)
    planet1.position = ring1.position - (ring_spec.rp - planet_spec.rp) * V3.j()
    planet1.orientation = ring1.orientation
    
    # Set the position and orientation of the second pair
    #ring2.position += gap_width * V3.k()
    #ring2.orientation = ring1.orientation
    #planet2.position = planet1.position + gap_width * V3.k()
    #planet2.orientation = planet1.orientation
    
    # Add the crank shaft. It is important that this is the first object added to the engine
    # as it is the object that is driven in the simulation
    """ crank_shaft_mesh = API.create_mesh(*MESH.create_cylinder(2 * ring_spec.rp, 2 * gear_width, 16))
    crank_shaft_position = V3.make(0, 0, -5 * gear_width)
    crank_shaft_orientation = Q.Rx(np.pi / 2)
    crank_shaft_name = API.add_object(engine, "crank_shaft", crank_shaft_mesh, crank_shaft_position, crank_shaft_orientation, material_name=material_name, density=density)
    """
    # Add the gears to the engine
    ring1_name = add_gear(ring1, "ring1")
    #ring2_name = add_gear(ring2, "ring2")
    
    # Add intermediate disks
    disk_mesh = MESH.Mesh(*MESH.create_cylinder(disk_radius, disk_width, 16))
    disk2_mesh = MESH.Mesh(*MESH.create_cylinder(disk_radius, disk_width, 16))
    
    disk_offset = 0.5 * (gear_width + disk_width) * V3.k() + eccentricity * V3.i()
    V1 = np.vstack([planet1.mesh.V, Q.rotate_array(Q.Rx(np.pi / 2), disk_mesh.V) + disk_offset])
    #V2 = np.vstack([planet2.mesh.V, Q.rotate_array(Q.Rx(np.pi / 2), disk2_mesh.V) + disk2_offset])
    T1 = np.vstack([planet1.mesh.T, disk_mesh.T + len(planet1.mesh.V)])
    #T2 = np.vstack([planet2.mesh.T, disk2_mesh.T + len(planet2.mesh.V)])
    
    # Actual disk position
    disk_position = planet1.position + 0.5 * (gear_width + disk_width) * V3.k() - eccentricity * V3.j()
    
    planet1.mesh = API.create_mesh(V1, T1)
    #planet2.mesh = API.create_mesh(V2, T2)
    
    planet1_name = add_gear(planet1, "planet1")
    #planet2_name = add_gear(planet2, "planet2")
    
    # Add connecting rod
    con_rod_height = 1.2 * ring1.spec.ra + clearance - disk_radius + eccentricity + ring1.spec.rp - planet1.spec.rp
    con_rod_mesh = API.create_mesh(*MESH.create_box(con_rod_width, con_rod_height, con_rod_width))
    con_rod_position = disk_position + disk_radius * V3.j() + 0.5 * con_rod_height * V3.j()
    con_rod_orientation = Q.identity()
    con_rod_name = API.add_object(engine, "connecting_rod", con_rod_mesh, con_rod_position, con_rod_orientation, material_name=material_name, density=density)
    
    # Add the piston
    piston_mesh = API.create_mesh(*MESH.create_cylinder(piston_radius, piston_height, 36))
    piston_position = (1.2 * ring1.spec.ra + clearance) * V3.j() + 0.5 * (gear_width + disk_width) * V3.k() + 0.5 * piston_height * V3.j()
    piston_name = API.add_object(engine, "piston", piston_mesh, piston_position, material_name=material_name, density=density)
    
    # Add the cylinder walls
    """ cylinder_radius = piston_radius
    cylinder_height = stroke_height + 2 * eccentricity + piston_height
    cylinder_thickness = 0.2
    cylinder_spacing = 0.01
    cylinder_center = piston_position - 0.5 * piston_height * V3.j() + 0.5 * cylinder_height * V3.j()
    cylinder_wall_directions = [V3.i(), V3.k(), -V3.i(), -V3.k()]
    cylinder_wall_positions = [
        cylinder_center + (cylinder_radius + cylinder_thickness / 2 + cylinder_spacing) * direction
        for direction in cylinder_wall_directions
    ]
    cylinder_wall_orientations = [
        Q.Ry(i * np.pi / 2)
        for i in range(4)
    ]
    for i, (position, orientation) in enumerate(zip(cylinder_wall_positions, cylinder_wall_orientations)):
        cylinder_wall_mesh = API.create_mesh(*MESH.create_box(cylinder_thickness, cylinder_height, 2 * (cylinder_radius + cylinder_thickness + cylinder_spacing)))
        API.add_object(engine, f"cylinder_wall_{i}", cylinder_wall_mesh, position, orientation, "fixed", material_name, density) """
    
    return
    
    # Add the ground
    ground_mesh = MESH.Mesh(*MESH.create_box(100, 1, 100))
    ground_position = V3.make(0, -100, 0)
    ground_orientation = Q.identity()
    ground_name = API.add_object(engine, "ground", ground_mesh, ground_position, ground_orientation, "fixed", material_name, density)

    # Create hinge joint between the ground and the crank shaft
    API.add_hinge(engine, ground_name, crank_shaft_name, crank_shaft_position, V3.k())
    
    # Create hinge joint between the crank shaft and the planet gears
    origin = disk_position #planet1.position + 0.5 * (planet2.position - planet1.position)
    API.add_hinge(engine, crank_shaft_name, planet1_name, origin, V3.k())
    #API.add_hinge(engine, crank_shaft_name, planet2_name, origin, V3.k())
    
    # Create hinge joint between the planet gears and the connecting rod
    origin1 = planet1.position + gear_width * V3.k() - eccentricity * V3.j()
    #origin2 = planet2.position - gear_width * V3.k() - eccentricity * V3.j()
    API.add_hinge(engine, planet1_name, con_rod_name, origin1, V3.k())
    #API.add_hinge(engine, planet2_name, connecting_rod_name, origin2, V3.k())
    
    # Create hinge joint between the connecting rod and the piston
    origin = piston_position - 0.5 * piston_height * V3.j()
    API.add_hinge(engine, con_rod_name, piston_name, origin, V3.k())
    
    # Create sliding joint for the piston
    origin = ring1.position
    API.add_sliding_joint(engine, ground_name, piston_name, origin, V3.j())
