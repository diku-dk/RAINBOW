import argparse
import datetime
import logging
import os
import time

# Profiling
import cProfile, pstats, io
from pstats import SortKey

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.scenes as SCENE

from rainbow.util.usd_manager import UsdManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", help="ID of the simulation", type=int, required=True)
    parser.add_argument("--max-cells", help="Maximum number of cells", type=int, default=1200)
    
    args = parser.parse_args()
    
    job_id: int = args.id
    max_cells: int = args.max_cells
    
    root_dir = f'{job_id}'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir, exist_ok=True)
    file_count = len(os.listdir(root_dir))
    
    animation_file: str = f'{root_dir}/animation_{file_count}.usda'
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=f"{root_dir}/simulation_{job_id}_{file_count}.log", encoding='utf-8', level=logging.DEBUG)
    logger.info(f"Starting simulation with ID: {job_id}")
    
    logger.info(f"Max cells: {max_cells}")
    
    pr = cProfile.Profile()
    pr.enable()
    
    engine = API.create_engine()
    
    cycles = 100
    steps_per_degree = 5
    
    revolutions_per_minute = 1000
    revolutions_per_second = revolutions_per_minute / 60
    
    seconds_per_cycle = 1 / revolutions_per_second * 3
    total_time = cycles * seconds_per_cycle
    
    steps_per_cycle = steps_per_degree * 1080
    step_size = seconds_per_cycle / steps_per_cycle
    
    logger.info(f"Revolutions per second: {revolutions_per_second}")
    logger.info(f"Revolutions per minute: {revolutions_per_minute}")
    logger.info(f"Seconds per cycle: {seconds_per_cycle}")
    logger.info(f"Steps per cycle: {steps_per_cycle}")
    logger.info(f"Step size: {step_size}")
    logger.info(f"Total time: {total_time}")

    angular_speed = 2 * np.pi * revolutions_per_second
    engine.params.time_step = step_size
    engine.params.driver_angular_velocity = angular_speed * V3.k()
    engine.params.sdf_max_cells = max_cells
    
    total_steps = int(total_time / engine.params.time_step)
    logger.info(f"Total steps: {total_steps}")
    
    logger.info(f"Animation will be saved to file: {animation_file}")
    
    SCENE.create_scene(engine, "engine")
    
    usd_manager = UsdManager(animation_file)
    usd_manager.initialize(engine)
    
    logger.info("Running simulation")
    total_duration = 0
    for step in range(total_steps):
        logger.info(f"Simulating step {step + 1}/{total_steps}")
        start = time.time()
        
        API.simulate(engine, engine.params.time_step)
        usd_manager.save_step(step, engine)
        
        end = time.time()
        duration = end - start
        total_duration += duration
        average_duration = total_duration / (step + 1)
        steps_remaining = total_steps - step - 1
        estimated_time_remaining = datetime.timedelta(seconds=steps_remaining * average_duration)
        
        logger.info(f"Step {step + 1}/{total_steps} took {duration:.2f} seconds (average: {average_duration:.2f} seconds) ETA: {estimated_time_remaining}")
    
    usd_manager.save()
    
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    logger.info(s.getvalue())
    
    logger.info("Simulation finished")


if __name__ == "__main__":
    main()
