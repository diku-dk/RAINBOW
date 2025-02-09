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
    
    m = 0.1
    z_ring = 79
    z_planet = 48
    face_width = 10 * m
    
    # Create the gears
    ring_spec = GEAR.GearSpec(m, z_ring, is_internal=True)
    planet_spec = GEAR.GearSpec(m, z_planet, is_internal=False)
    
    eccentricity = 0.25 * planet_spec.rp
    gap_width = 4 * face_width
    
    gear_factory = GEAR.GearFactory()
    
    # Add gears
    ring1 = gear_factory.create_gear(ring_spec, face_width)
    ring2 = gear_factory.create_gear(ring_spec, face_width)
    planet1 = gear_factory.create_gear(planet_spec, face_width)
    planet2 = gear_factory.create_gear(planet_spec, face_width)
    
    # Set the position and orientation of the first pair
    ring1.orientation = Q.Rz(-np.pi / 2)
    planet1.position = ring1.position - (ring_spec.rp - planet_spec.rp) * V3.j()
    planet1.orientation = ring1.orientation #MATING.compute_gear_orientation(ring_spec, planet_spec, -np.pi/2 - 2 * np.pi)
    
    # Set the position and orientation of the second pair
    ring2.position += gap_width * V3.k()
    ring2.orientation = ring1.orientation
    planet2.position = planet1.position + gap_width * V3.k()
    planet2.orientation = planet1.orientation
    
    # Add the crank shaft. It is important that this is the first object added to the engine
    # as it is the object that is driven in the simulation
    crank_shaft_mesh = API.create_mesh(*MESH.create_cylinder(2 * ring_spec.rp, 2 * face_width, 16))
    crank_shaft_position = V3.make(0, 0, -2.1 * face_width)
    crank_shaft_orientation = Q.Rx(np.pi / 2)
    crank_shaft_name = API.add_object(engine, "crank_shaft", crank_shaft_mesh, crank_shaft_position, crank_shaft_orientation, material_name=material_name, density=density)

    # Add the gears to the engine
    ring1_name = add_gear(ring1, "ring1")
    ring2_name = add_gear(ring2, "ring2")
    
    # Add intermediate disks
    disk_radius = 0.5 * planet_spec.rp
    disk1_mesh = MESH.Mesh(*MESH.create_cylinder(disk_radius, 0.8 * face_width, 16))
    disk2_mesh = MESH.Mesh(*MESH.create_cylinder(disk_radius, 0.8 * face_width, 16))
    
    disk1_offset = face_width * V3.k() + eccentricity * V3.i()
    disk2_offset = -face_width * V3.k() + eccentricity * V3.i()
    V1 = np.vstack([planet1.mesh.V, Q.rotate_array(Q.Rx(np.pi / 2), disk1_mesh.V) + disk1_offset])
    V2 = np.vstack([planet2.mesh.V, Q.rotate_array(Q.Rx(np.pi / 2), disk2_mesh.V) + disk2_offset])
    T1 = np.vstack([planet1.mesh.T, disk1_mesh.T + len(planet1.mesh.V)])
    T2 = np.vstack([planet2.mesh.T, disk2_mesh.T + len(planet2.mesh.V)])
    
    planet1.mesh = API.create_mesh(V1, T1)
    planet2.mesh = API.create_mesh(V2, T2)
    
    planet1_name = add_gear(planet1, "planet1")
    planet2_name = add_gear(planet2, "planet2")
    
    # Add connecting rod
    connecting_rod_height = 2.5 * ring_spec.rp
    connecting_rod_mesh = API.create_mesh(*MESH.create_box(face_width, connecting_rod_height, face_width))
    connecting_rod_position = planet1.position + 0.5 * (planet2.position - planet1.position) + 0.4 * connecting_rod_height * V3.j() - eccentricity * V3.j()
    connecting_rod_orientation = Q.identity()
    connecting_rod_name = API.add_object(engine, "connecting_rod", connecting_rod_mesh, connecting_rod_position, connecting_rod_orientation, material_name=material_name, density=density)
    
    # Add the ground
    ground_mesh = MESH.Mesh(*MESH.create_box(100, 1, 100))
    ground_position = V3.make(0, -2.2 * ring_spec.rp, 0)
    ground_orientation = Q.identity()
    ground_name = API.add_object(engine, "ground", ground_mesh, ground_position, ground_orientation, "fixed", material_name, density)

    # Create hinge joint between the ground and the crank shaft
    API.add_hinge(engine, ground_name, crank_shaft_name, crank_shaft_position, V3.k())
    
    # Create hinge joint between the crank shaft and the planet gears
    origin = planet1.position + 0.5 * (planet2.position - planet1.position)
    API.add_hinge(engine, crank_shaft_name, planet1_name, origin, V3.k())
    API.add_hinge(engine, crank_shaft_name, planet2_name, origin, V3.k())
    
    # Create hinge joint between the planet gears and the connecting rod
    origin1 = planet1.position + face_width * V3.k() - eccentricity * V3.j()
    origin2 = planet2.position - face_width * V3.k() - eccentricity * V3.j()
    API.add_hinge(engine, planet1_name, connecting_rod_name, origin1, V3.k())
    API.add_hinge(engine, planet2_name, connecting_rod_name, origin2, V3.k())
