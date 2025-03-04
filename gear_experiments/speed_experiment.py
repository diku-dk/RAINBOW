import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import polyscope as ps

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sdf_resolution', type=int, help='The resolution of the SDF grid')
    parser.add_argument('--headless', action='store_true', help='Run the simulation in headless mode')
    parser.add_argument('--steps', type=int, default=2, help='The number of steps to run the simulation for')
    parser.add_argument('--z1', type=int, help='Number of teeth on the first gear')
    parser.add_argument('--z2', type=int, help='Number of teeth on the second gear')
    
    args = parser.parse_args()
    sdf_resolution: int = args.sdf_resolution
    headless: bool = args.headless
    steps: int = args.steps
    z1: int = args.z1
    z2: int = args.z2
    
    logging.basicConfig(level=logging.INFO)

    engine = API.create_engine()
    engine.params.time_step = 0.001
    engine.params.driver_angular_velocity = 0.1 * V3.k()
    engine.params.sdf_min_cells = sdf_resolution
    engine.params.sdf_max_cells = sdf_resolution
    engine.params.resolution = sdf_resolution
    
    app = GearApp(engine, f'{FILE_DIR}/speed_{z1}_{z2}', headless)

    factory = GEAR.generators.GearFactory()

    m = 1
    face_width = 10 * m
    
    spec1 = GEAR.types.GearSpec(m, z1)
    spec2 = GEAR.types.GearSpec(m, z2)
    
    gear1 = factory.create_gear(spec1, face_width, subdivisions=0)
    gear2 = factory.create_gear(spec2, face_width, subdivisions=0)
    gear2.position = (gear1.spec.rp + gear2.spec.rp) * V3.i()
    gear2.orientation = GEAR.mating.compute_gear_orientation(gear1.spec, gear2.spec, 0)
    
    gear1_name = API.add_object(engine, 'gear1', gear1.mesh, gear1.position, gear1.orientation)
    gear2_name = API.add_object(engine, 'gear2', gear2.mesh, gear2.position, gear2.orientation)
    
    max_ra = max(gear1.spec.ra, gear2.spec.ra)
    ground_name = PROC.create_ground(
        engine,
        (-1.5 * max_ra) * V3.j() + (gear2.position - gear1.position) / 2,
        Q.identity(),
        width=max_ra * 3,
        height=5 * m,
    )
    
    API.add_hinge(engine, gear1_name, ground_name, gear1.position, V3.k())
    API.add_hinge(engine, gear2_name, ground_name, gear2.position, V3.k())

    gear1_spin = []
    gear2_spin = []
    screenshot_taken = False
    def callback(engine: TYPES.Engine):
        nonlocal screenshot_taken
        if not screenshot_taken:
            ps.screenshot(f'{FILE_DIR}/speed/screenshot_{z1}_{z2}.png')
            screenshot_taken = True
        gear1_spin.append(API.get_spin(engine, gear1_name))
        gear2_spin.append(API.get_spin(engine, gear2_name))

    app.run(steps, callback=callback)

    gear1_spin = np.array(gear1_spin)
    gear2_spin = np.array(gear2_spin)
    
    plot_speeds(gear1_spin, gear2_spin, z1, z2)
    
    gear1_mean_speed = np.mean(np.linalg.norm(gear1_spin[100:], axis=1))
    gear2_mean_speed = np.mean(np.linalg.norm(gear2_spin[100:], axis=1))
    
    print(f'Gear 1 mean speed: {gear1_mean_speed}')
    print(f'Gear 2 mean speed: {gear2_mean_speed}')
    print(f'Gear 1 speed ratio: {gear1_mean_speed / gear2_mean_speed}')


def plot_speeds(gear1_spins, gear2_spins, z1, z2):
    plt.figure(figsize=(6, 4), dpi=300)
    
    plt.plot(np.linalg.norm(gear1_spins, axis=1), label='Gear 1')
    plt.plot(np.linalg.norm(gear2_spins, axis=1), label='Gear 2')

    plt.title(f'Gear Angular Speeds (Z1={z1}, Z2={z2})')
    plt.xlabel('Steps')
    plt.ylabel('Angular Speed (rad/s)')
    plt.legend()
    plt.savefig(f'{FILE_DIR}/speed/planet_speeds_{z1}_{z2}.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
