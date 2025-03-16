import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import polyscope as ps

import pickle

# Add the parent directory to the path to import from rainbow
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(FILE_DIR))

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.geometry.surface_mesh as MESH
import rainbow.simulators.prox_rigid_bodies.types as TYPES
import rainbow.procedural.gears as GEAR
import rainbow.simulators.prox_rigid_bodies.scenes as PROC

from gear_app import GearApp


plt.rc('font', size=12)
plt.rc('font', family='serif')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sdf_resolution', type=int, help='The resolution of the SDF grid')
    parser.add_argument('--headless', action='store_true', help='Run the simulation in headless mode')
    parser.add_argument('--steps', type=int, default=2, help='The number of steps to run the simulation for')
    
    args = parser.parse_args()
    sdf_resolution: int = args.sdf_resolution
    headless: bool = args.headless
    steps: int = args.steps
    
    logging.basicConfig(level=logging.INFO)

    engine = API.create_engine()
    engine.params.time_step = 0.001
    engine.params.driver_angular_velocity = 2 * V3.k()
    engine.params.sdf_min_cells = 64
    engine.params.sdf_max_cells = sdf_resolution
    engine.params.resolution = sdf_resolution
    
    app = GearApp(engine, f'{FILE_DIR}/backlash/PlanetaryGear_{sdf_resolution}', headless)

    factory = GEAR.generators.GearFactory()

    m = 1
    face_width = 10 * m
    
    z_sun = 31
    z_planet = 17
    z_ring = z_sun + 2 * z_planet
    planetary_spec = GEAR.types.PlanetaryGearSpec(m, z_sun, z_planet, z_ring, N_planet=3, helix_angle=None)
    planetary_gear = factory.create_planetary_gear(planetary_spec, face_width, subdivisions=3)

    sun_gear = planetary_gear.sun_gear
    planet_gears = planetary_gear.planet_gears
    ring_gear = planetary_gear.ring_gear

    sun_name = API.add_object(engine, 'sun_gear', sun_gear.mesh, sun_gear.position, sun_gear.orientation)

    planet_names = []
    for i, gear in enumerate(planet_gears):
        planet_name = API.add_object(engine, f'planet_gear_{i}', gear.mesh, gear.position, gear.orientation)
        planet_names.append(planet_name)

    ring_name = API.add_object(engine, 'ring_gear', ring_gear.mesh, ring_gear.position, ring_gear.orientation, body_type='fixed')

    flywheel_radius = ring_gear.spec.rd * 0.95
    flywheel_width = face_width / 8
    V, T = MESH.create_cylinder(flywheel_radius, flywheel_width, 18)
    flywheel_mesh = API.create_mesh(V, T)
    flywheel_position = ring_gear.position - face_width * V3.k()
    flywheel_orientation = Q.Rx(np.pi / 2)
    flywheel_name = API.add_object(engine, 'flywheel', flywheel_mesh, flywheel_position, flywheel_orientation)
    
    ground_name = PROC.create_ground(
        engine,
        (-1.5 * planetary_gear.ring_gear.spec.rd) * V3.j(),
        Q.identity(),
        width=planetary_gear.ring_gear.spec.rd * 3,
        height=5 * m,
    )

    API.add_hinge(engine, ground_name, flywheel_name, flywheel_position, V3.k())

    API.add_hinge(engine, ground_name, sun_name, sun_gear.position, V3.k())
    for planet_name, planet_gear in zip(planet_names, planet_gears):
        API.add_hinge(engine, flywheel_name, planet_name, planet_gear.position, V3.k())

    sun_velocities = []
    planet_velocities = []
    ring_velocities = []
    screenshot_taken = False
    def callback(engine: TYPES.Engine):
        nonlocal screenshot_taken
        if not screenshot_taken:
            ps.screenshot(f'{FILE_DIR}/backlash/screenshot_{sdf_resolution}.png')
            screenshot_taken = True
        sun_velocities.append(API.get_spin(engine, sun_name))
        planet_velocities.append([API.get_spin(engine, name) for name in planet_names])
        ring_velocities.append(API.get_spin(engine, ring_name))

    app.run(steps, callback=callback)

    sun_velocities = np.array(sun_velocities)
    planet_velocities = np.array(planet_velocities).reshape(-1, 3, 3)
    ring_velocities = np.array(ring_velocities)
    
    pickle.dump((sdf_resolution, sun_velocities, planet_velocities), open(f'{FILE_DIR}/backlash/velocities_{sdf_resolution}.pkl', 'wb'))
    
    plot_speeds(planet_names, planet_velocities, sun_velocities, sdf_resolution, include_sun=True)
    plot_speeds(planet_names, planet_velocities, sun_velocities, sdf_resolution, include_sun=False)
    plot_speeds(planet_names, planet_velocities, sun_velocities, sdf_resolution, include_sun=True, only_z_axis=True)
    plot_speeds(planet_names, planet_velocities, sun_velocities, sdf_resolution, include_sun=False, only_z_axis=True)


def plot_speeds(planet_names, planet_velocities, sun_velocities, sdf_resolution, include_sun=False, only_z_axis=False):
    if only_z_axis:
        planet_speeds = np.abs(planet_velocities[:, :, 2])
    else:
        planet_speeds = np.linalg.norm(planet_velocities, axis=2)
    
    
    plt.figure(figsize=(6, 4), dpi=300)
    for i in range(len(planet_names)):
        plt.plot(planet_speeds[:, i], label=f'Planet {i}')
    
    if include_sun:
        plt.plot(np.linalg.norm(sun_velocities, axis=1), label='Sun')

    plt.title(f'Planet Speeds (SDF Grid Size: {sdf_resolution}x{sdf_resolution}x{sdf_resolution})')
    plt.xlabel('Steps')
    plt.ylabel('Angular Speed (rad/s)')
    plt.legend()
    plt.savefig(f'{FILE_DIR}/backlash/planet_speeds_{sdf_resolution}{"_with_sun" if include_sun else ""}{"_onlyz" if only_z_axis else ""}.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
