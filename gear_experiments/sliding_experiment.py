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
    parser.add_argument('--steps', type=int, default=50, help='The number of steps to run the simulation for')
    parser.add_argument('--z1', type=int, default=20, help='Number of teeth on the first gear')
    parser.add_argument('--z2', type=int, default=20, help='Number of teeth on the second gear')
    
    args = parser.parse_args()
    sdf_resolution: int = args.sdf_resolution
    headless: bool = args.headless
    steps: int = args.steps
    z1: int = args.z1
    z2: int = args.z2
    
    logging.basicConfig(level=logging.INFO)

    engine = API.create_engine()
    engine.params.time_step = 0.001
    engine.params.driver_angular_velocity = V3.k()
    engine.params.sdf_min_cells = 64
    engine.params.sdf_max_cells = sdf_resolution
    engine.params.resolution = sdf_resolution
    
    app = GearApp(engine, f'{FILE_DIR}/sliding_{z1}_{z2}', headless)

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
    
    API.create_damping_force(engine=engine, force_name="air", alpha=0.01, beta=0.01)
    for body in engine.bodies.values():
        API.connect_force(engine=engine, body_name=body.name, force_name="air")

    all_tangential_velocities = []
    all_relative_velocities = []
    all_normal_velocities = []

    screenshot_taken = False
    def callback(engine: TYPES.Engine):
        nonlocal screenshot_taken
        if not screenshot_taken:
            ps.screenshot(f'{FILE_DIR}/sliding/screenshot_{z1}_{z2}.png')
            screenshot_taken = True
        
        print(f'Number of contact points: {len(engine.contact_points)}')
        
        tangential_velocities = np.zeros((len(engine.contact_points), 1))
        relative_velocities = np.zeros((len(engine.contact_points), 1))
        normal_velocities = np.zeros((len(engine.contact_points), 1))
        for i, cp in enumerate(engine.contact_points):
            r1 = cp.p - cp.bodyA.r
            r2 = cp.p - cp.bodyB.r
            v1 = np.cross(cp.bodyA.w, r1)
            v2 = np.cross(cp.bodyB.w, r2)
            
            n = cp.n / np.linalg.norm(cp.n)
            
            v_rel = v2 - v1
            v_norm_component = np.dot(v_rel, n)
            v_normal = v_norm_component * n
            v_tangent = v_rel - v_normal
            tangential_velocities[i] = np.linalg.norm(v_tangent)
            relative_velocities[i] = np.linalg.norm(v_rel)
            normal_velocities[i] = np.linalg.norm(v_normal)
        
        all_tangential_velocities.append(tangential_velocities)
        all_relative_velocities.append(relative_velocities)
        all_normal_velocities.append(normal_velocities)

    app.run(steps, callback=callback)

    plot_speeds(all_tangential_velocities, all_relative_velocities, all_normal_velocities)


def plot_speeds(all_tangential_velocities, all_relative_velocities, all_normal_velocities):
    plt.figure(figsize=(6, 4), dpi=300)

    for i in range(len(all_tangential_velocities)):
        if len(all_tangential_velocities[i]) == 0:
            continue
        xs = np.full(len(all_tangential_velocities[i]), i, dtype=int)
        ys = all_tangential_velocities[i]
        plt.scatter(xs, ys)

    plt.title(f'Tangential Velocities')
    plt.xlabel('Steps')
    plt.ylabel('Tangential Velocity')
    plt.savefig(f'{FILE_DIR}/sliding/tangential_velocities.png')
    
    plt.figure(figsize=(6, 4), dpi=300)
    
    for i in range(len(all_relative_velocities)):
        if len(all_relative_velocities[i]) == 0:
            continue
        xs = np.full(len(all_relative_velocities[i]), i, dtype=int)
        ys = np.linalg.norm(all_relative_velocities[i], axis=1)
        plt.scatter(xs, ys)
    
    plt.title(f'Relative Velocities')
    plt.xlabel('Steps')
    plt.ylabel('Relative Velocity')
    plt.savefig(f'{FILE_DIR}/sliding/relative_velocities.png')
    
    plt.figure(figsize=(6, 4), dpi=300)
    
    for i in range(len(all_normal_velocities)):
        if len(all_normal_velocities[i]) == 0:
            continue
        xs = np.full(len(all_normal_velocities[i]), i, dtype=int)
        ys = np.linalg.norm(all_normal_velocities[i], axis=1)
        plt.scatter(xs, ys)
    
    plt.title(f'Normal Velocities')
    plt.xlabel('Steps')
    plt.ylabel('Normal Velocity')
    plt.savefig(f'{FILE_DIR}/sliding/normal_velocities.png')


if __name__ == '__main__':
    main()
