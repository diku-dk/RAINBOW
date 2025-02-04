"""
This script contains code to create gear shapes and gear mechanisms.
"""

import numpy as np

import rainbow.math.involute as INV
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.geometry.surface_mesh as MESH
from rainbow.simulators.prox_rigid_bodies.types import Engine

import rainbow.procedural.gears.generators as GEAR
import rainbow.procedural.gears.mating as MATING


def create_gear_train(engine: Engine,
                      N: int,
                      density: float = 1.0,
                      material_name: str = "default"
                      ) -> list[str]:
    """
    This function creates a sequence of N gears.

    All gears are paired up to form a chain of gears.

    :param engine:                    The engine that will be used to create the dry stone rigid bodies in.
    :param N:                         The number of gears to create in the scene.
    :param density:                   The mass density to use for all the rigid bodies.
    :param material_name:             The material name to use for all the rigid bodies that are created.
    :return:                          A list with the names of all the rigid bodies that were created.
    """
    body_names = []
    gear_names = []

    # Create and place all the gears
    q_m2w = Q.Rx(-np.pi / 2)  # Needed to change the z-up direction to a y-up direction.

    face_width = 10.0  # Width of the gear.
    
    gear_factory = GEAR.GearFactory()
    
    gear_spec1 = GEAR.GearSpec(2, 12)
    gear_spec2 = GEAR.GearSpec(2, 17)
    
    gear_specs = [gear_spec1, gear_spec2]
    
    gear1, gear2 = [
        gear_factory.create_gear(spec, face_width, subdivisions=3) for spec in gear_specs
    ]
    
    gear2.position = MATING.compute_gear_position(gear_spec1, gear_spec2, 0.0)
    gear2.orientation = MATING.compute_gear_orientation(gear_spec1, gear_spec2, 0.0)
    
    for i, gear in enumerate([gear1, gear2]):
        shape_name = API.generate_unique_name("shape")
        body_name = API.generate_unique_name("body")

        body_names.append(body_name)
        gear_names.append(shape_name)

        API.create_shape(engine, shape_name, gear.mesh)

        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)
        
        r = API.get_position(engine, body_name)

        r_m = gear.position  # Model space position of gear (z-up).
        q_m = gear.orientation  # Model space orientation of gear (z-up).
        r_w = Q.rotate(q_m2w, r_m)  # World position of gear (y-up).
        q_w = Q.prod(q_m2w, q_m)  # World orientation of gear (y-up).
        
        API.set_position(engine, body_name, r_w, True)
        API.set_orientation(engine, body_name, q_w, True)
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    # Create a fixed object in the world
    body_name = create_ground(engine, face_width, density, material_name)
    body_names.append(body_name)

    # Create hinge-joints between gears and fixed object
    parent_name = body_name
    for i in range(len(gear_specs)):
        child_name = body_names[i]
        hinge_name = parent_name + "_" + child_name
        API.create_hinge(engine, hinge_name)
        origin = API.get_position(engine, child_name) #- V3.make(0.0, face_width/2.0, 0.0)
        API.set_hinge(
            engine=engine,
            hinge_name=hinge_name,
            parent_name=parent_name,
            child_name=child_name,
            origin= origin,
            axis= V3.j(),
            mode="world"
        )

    return body_names


def create_planetary_gear(engine: Engine,
                          density: float = 1.0,
                          material_name: str = "default"
                          ) -> list[str]:
    
    body_names = []
    gear_names = []
    
    # Create and place all the gears
    q_m2w = Q.Rx(-np.pi / 2)  # Needed to change the z-up direction to a y-up direction.
    
    face_width = 10.0  # Width of the gear.
    
    gear_factory = GEAR.GearFactory()
    
    m = 2
    z_sun = 33
    z_planet = 18
    z_ring = z_sun + 2 * z_planet
    
    planetary_spec = GEAR.PlanetaryGearSpec(m, z_sun, z_planet, z_ring)
    
    planetary_gear = gear_factory.create_planetary_gear(planetary_spec, face_width, subdivisions=3)
    
    sun_gear = planetary_gear.sun_gear
    planet_gears = planetary_gear.planet_gears
    ring_gear = planetary_gear.ring_gear
    
    gears = [sun_gear] + planet_gears + [ring_gear]
    
    for i, gear in enumerate(gears):
        shape_name = API.generate_unique_name("shape")
        body_name = API.generate_unique_name("body")
        
        body_names.append(body_name)
        gear_names.append(shape_name)
        
        API.create_shape(engine, shape_name, gear.mesh)
        
        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)
        
        r_m = gear.position # Model space position of gear (z-up).
        q_m = gear.orientation # Model space orientation of gear (z-up).
        r_w = Q.rotate(q_m2w, r_m) # World position of gear (y-up).
        q_w = Q.prod(q_m2w, q_m) # World orientation of gear (y-up).
        
        API.set_position(engine, body_name, r_w, True)
        API.set_orientation(engine, body_name, q_w, True)
        if i == len(gears) - 1:
            API.set_body_type(engine, body_name, "fixed")
        else:
            API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)
    
    # Create a fixed object in the world
    body_name = create_ground(engine, face_width, density, material_name)
    body_names.append(body_name)
    
    # Create hinge-joints between gears and fixed object
    parent_name = body_name
    child_name = body_names[0]
    hinge_name = parent_name + "_" + child_name
    API.create_hinge(engine, hinge_name)
    origin = API.get_position(engine, child_name)
    API.set_hinge(
        engine=engine,
        hinge_name=hinge_name,
        parent_name=parent_name,
        child_name=child_name,
        origin=origin,
        axis=V3.j(),
        mode="world"
    )
    
    fly_wheel_shape_name = API.generate_unique_name("fly_wheel")
    fly_wheel_body_name = API.generate_unique_name("fly_wheel")
    
    V, T = MESH.create_cylinder(ring_gear.spec.ra, 2.0, 16)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, fly_wheel_shape_name, mesh)
    
    API.create_rigid_body(engine, fly_wheel_body_name)
    API.connect_shape(engine, fly_wheel_body_name, fly_wheel_shape_name)
    
    r = V3.make(0.0, -5.0 - face_width, 0.0)
    q = Q.identity()
    
    API.set_position(engine, fly_wheel_body_name, r, True)
    API.set_orientation(engine, fly_wheel_body_name, q, True)
    API.set_body_type(engine, fly_wheel_body_name, "free")
    API.set_body_material(engine, fly_wheel_body_name, material_name)
    API.set_mass_properties(engine, fly_wheel_body_name, density)
    
    body_names.append(fly_wheel_body_name)
    
    # Create hinge-joints between gears and fixed object
    child_name = fly_wheel_body_name
    hinge_name = parent_name + "_" + child_name
    API.create_hinge(engine, hinge_name)
    origin = API.get_position(engine, child_name)
    API.set_hinge(
        engine=engine,
        hinge_name=hinge_name,
        parent_name=parent_name,
        child_name=child_name,
        origin=origin,
        axis=V3.j(),
        mode="model"
    )
    
    parent_name = child_name
    
    for child_name in body_names[1:]:
        hinge_name = parent_name + "_" + child_name
        API.create_hinge(engine, hinge_name)
        origin = API.get_position(engine, child_name)
        API.set_hinge(
            engine=engine,
            hinge_name=hinge_name,
            parent_name=parent_name,
            child_name=child_name,
            origin=origin,
            axis=V3.j(),
            mode="world"
        )
    

def create_ground(engine: Engine, face_width: float, density: float, material_name: str) -> None:
    shape_name = API.generate_unique_name("ground_shape")

    V, T = MESH.create_box(200.0, 1.0, 200.0)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("ground_body")
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(0.0, -10.5 - face_width, 0.0)
    q = Q.identity()

    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)

    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)
    return body_name
