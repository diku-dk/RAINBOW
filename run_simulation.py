import argparse
import time
import tqdm

import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.scenes as SCENE

from rainbow.util.usd_manager import UsdManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", help="Scene to simulate")
    parser.add_argument("--animation-file", help="File to save the animation", default="animation.usda")
    parser.add_argument("--total-time", help="Length of the simulation in seconds", type=float, default=1.0)
    parser.add_argument("--time-step", help="Time step of the simulation in seconds", type=float)
    
    args = parser.parse_args()
    
    scene = args.scene
    animation_file = args.animation_file
    total_time = args.total_time
    time_step = args.time_step
    
    print(f"Total time: {total_time}")
    
    engine = API.create_engine()
    if time_step is not None:
        print(f"Time step: {time_step}")
        engine.params.time_step = time_step
    
    total_steps = int(total_time / engine.params.time_step)
    print(f"Total steps: {total_steps}")
    
    print(f"Animation will be saved to file: {animation_file}")
    
    print(f"Creating scene: {scene}")
    SCENE.create_scene(engine, scene)
    
    usd_manager = UsdManager(animation_file)
    usd_manager.initialize(engine)
    
    print("Running simulation")
    for step in tqdm.tqdm(range(total_steps)):
        API.simulate(engine, engine.params.time_step)
        usd_manager.save_step(step, engine)
    
    usd_manager.save()
    
    print("Simulation finished")


if __name__ == "__main__":
    main()
