
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
    
    def add_object(name: str, mesh: MESH.Mesh, position: np.ndarray, orientation: np.ndarray, body_type: str) -> str:
        shape_name = API.generate_unique_name(name + "_shape")
        body_name = API.generate_unique_name(name + "_body")
        API.create_shape(engine, shape_name, mesh)
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)
        
        API.set_position(engine, body_name, position, True)
        API.set_orientation(engine, body_name, orientation, True)
        API.set_body_type(engine, body_name, body_type)
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)
        return body_name
    
    def add_gear(gear: GEAR.Gear, name: str):
        return add_object(name, gear.mesh, gear.position, gear.orientation, "fixed" if gear.spec.is_internal else "free")
    
    def add_hinge(parent_name: str, child_name, axis: np.ndarray, origin: np.ndarray) -> str:
        hinge_name = parent_name + "_" + child_name
        API.create_hinge(engine, hinge_name)
        API.set_hinge(
            engine,
            hinge_name,
            parent_name,
            child_name,
            origin,
            axis,
            "world",
        )
        return hinge_name
    
    m = 1
    z_ring = 79
    z_planet = 48
    face_width = 10
    
    # Create the gears
    ring_spec = GEAR.GearSpec(m, z_ring, is_internal=True)
    planet_spec = GEAR.GearSpec(m, z_planet, is_internal=False)
    
    eccentricity = 0.25 * planet_spec.rp
    gap_width = 4 * face_width
    
    gear_factory = GEAR.GearFactory()
    
    # Add first pair of gears
    ring1 = gear_factory.create_gear(ring_spec, face_width)
    planet1 = gear_factory.create_gear(planet_spec, face_width)
    
    # Set the position and orientation of the planet gear
    planet1.position += (ring_spec.rp - planet_spec.rp) * V3.make(0, -1, 0)
    planet1.orientation = MATING.compute_gear_orientation(ring_spec, planet_spec, -np.pi/2)
    
    # Add second pair of gears
    ring2 = gear_factory.create_gear(ring_spec, face_width)
    planet2 = gear_factory.create_gear(planet_spec, face_width)
    
    # Set the position and orientation of the gears
    ring2.position += gap_width * V3.k()
    planet2.position = planet1.position + gap_width * V3.k()
    planet2.orientation = planet1.orientation
    
    # Add the crank shaft. It is important that this is the first object added to the engine
    # as it is the object that is driven in the simulation
    crank_shaft_mesh = API.create_mesh(*MESH.create_cylinder(2 * ring_spec.rp, 2 * face_width, 16))
    crank_shaft_position = V3.make(0, 0, -2.1 * face_width)
    crank_shaft_orientation = Q.Rx(np.pi / 2)
    crank_shaft_name = add_object("crank_shaft", crank_shaft_mesh, crank_shaft_position, crank_shaft_orientation, "free")

    # Add the gears to the engine
    ring1_name = add_gear(ring1, "ring1")
    planet1_name = add_gear(planet1, "planet1")
    ring2_name = add_gear(ring2, "ring2")
    planet2_name = add_gear(planet2, "planet2")
    
    # Add interemdiate disks
    disk_radius = 0.5 * planet_spec.rp
    disk1_mesh = API.create_mesh(*MESH.create_cylinder(disk_radius, 0.8 * face_width, 16))
    disk2_mesh = API.create_mesh(*MESH.create_cylinder(disk_radius, 0.8 * face_width, 16))
    disk1_position = planet1.position + face_width * V3.k() - eccentricity * V3.j()
    disk2_position = disk1_position + 2 * face_width * V3.k()
    disk_orientation = Q.Rx(np.pi / 2)
    disk1_name = add_object("disk1", disk1_mesh, disk1_position, disk_orientation, "free")
    disk2_name = add_object("disk2", disk2_mesh, disk2_position, disk_orientation, "free")
    
    # Add connecting rod
    connecting_rod_height = 2.5 * ring_spec.rp
    connecting_rod_mesh = API.create_mesh(*MESH.create_box(face_width, connecting_rod_height, face_width))
    connecting_rod_position = disk1_position + 0.5 * connecting_rod_height * V3.j() + face_width * V3.k() - eccentricity * V3.j()
    connecting_rod_orientation = Q.identity()
    connecting_rod_name = add_object("connecting_rod", connecting_rod_mesh, connecting_rod_position, connecting_rod_orientation, "free")
    
    # Add the ground
    ground_mesh = MESH.Mesh(*MESH.create_box(100, 1, 100))
    ground_position = V3.make(0, -2.2 * ring_spec.rp, 0)
    ground_orientation = Q.identity()
    ground_name = add_object("ground", ground_mesh, ground_position, ground_orientation, "fixed")

    # Create hinge joint between the ground and the crank shaft
    add_hinge(ground_name, crank_shaft_name, V3.k(), crank_shaft_position)
    
    # Create hinge joint between the crank shaft and the planet gears
    origin = API.get_position(engine, planet1_name) - V3.make(0, planet_spec.rp - ring_spec.rp, 0)
    add_hinge(crank_shaft_name, planet1_name, V3.k(), origin)
    add_hinge(crank_shaft_name, planet2_name, V3.k(), origin)

    # Create hinge joint between the planet gears and the intermediate disks
    origin1 = planet1.position
    origin2 = planet2.position
    add_hinge(planet1_name, disk1_name, V3.k(), origin1)
    add_hinge(planet2_name, disk2_name, V3.k(), origin2)
    
    # Create hinge joint between the intermediate disks and the connecting rod
    origin1 = disk1_position
    origin2 = disk2_position
    add_hinge(disk1_name, connecting_rod_name, V3.k(), origin1)
    add_hinge(disk2_name, connecting_rod_name, V3.k(), origin2)
