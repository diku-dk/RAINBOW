from __future__ import annotations

import logging
import numpy as np
import os
import polyscope as ps
import sys

from typing import Callable

# Add the parent directory to the path to import from rainbow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rainbow.math.quaternion as Q
import rainbow.math.vector3 as V3
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.types as TYPES

from rainbow.util.usd_manager import UsdManager


class GearApp:
    def __init__(
        self,
        engine: TYPES.Engine,
        usd_path: str | None = None,
        headless: bool = False,
    ):
        self.logger = logging.getLogger(f'{__name__}.GearApp')
        self.logger.info("Initializing GearApp")
        
        self.engine = engine
        self.time_step = self.engine.params.time_step
        self.usd_manager = UsdManager(usd_path if usd_path.endswith('.usd') else usd_path + '.usd') if usd_path is not None else None
        self.headless = headless
        
        self.steps = None
        self.custom_callback = None
        
    
    def run(self, steps: int, callback: Callable[[TYPES.Engine], None] | None = None):
        logger = logging.getLogger(f'{__name__}.GearApp.run')
        logger.info(f'Running GearApp for {steps} steps')
        
        self.steps = steps
        self.custom_callback = callback
        
        self.step = 0
        
        if self.usd_manager is not None:
            self.usd_manager.initialize(self.engine)
        
        if self.headless:
            logger.info("Running in headless mode")
            # Manually call the callback in headless mode
            for _ in range(steps):
                self.callback()
        else:
            self._ps_setup()
            ps.show(self.steps + 1)
    
    def callback(self):
        logger = logging.getLogger(f'{__name__}.GearApp.callback')
        
        if self.step > 0:
            logger.info(f'Running simulation step {self.step}')
            API.simulate(self.engine, self.time_step)
            logger.info(f'Simulation step {self.step} complete')
        
            if self.usd_manager is not None:
                self.usd_manager.save_step(self.step, self.engine)

        self.usd_manager.save()
        
        if not self.headless:
            self._ps_update()
        
        if self.custom_callback is not None:
            self.custom_callback(self.engine)
        
        self.step += 1

    def _ps_setup(self):
        logger = logging.getLogger(f'{__name__}.GearApp._ps_setup')
        logger.info('Setting up polyscope')

        ps.set_up_dir('y_up')
        ps.init()
        ps.set_ground_plane_mode('none')
        look_dir = np.array([1, 1, 10], dtype=np.float64)
        look_dir /= np.linalg.norm(look_dir)
        ps.look_at(15 * look_dir, (0, 0, 0))
        ps.set_user_callback(self.callback)

        for body in self.engine.bodies.values():
            color = V3.make(1, 0.1, 0.1) if not body.is_fixed else V3.make(0.1, 0.1, 1)
            ps.register_surface_mesh(
                body.name,
                body.shape.mesh.V,
                body.shape.mesh.T,
                smooth_shade=False,
                color=color,
            )
        
        self._ps_update()
    
    def _ps_update(self):
        for body in self.engine.bodies.values():
            transformation_matrix = np.eye(4, dtype=np.float64)
            transformation_matrix[:3, :3] = Q.to_matrix(body.q)
            transformation_matrix[:3, 3] = body.r
            ps.get_surface_mesh(body.name).set_transform(transformation_matrix)
