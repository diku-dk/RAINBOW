
import logging

import rainbow.simulators.prox_rigid_bodies.types as TYPES
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.simulators.prox_rigid_bodies.scenes as PROC
import rainbow.math.quaternion as Q
import rainbow.math.vector3 as V3

def create_simple_gear(engine: TYPES.Engine):
    logger = logging.getLogger("main.setup_scene.create_simple_gear_scene")
    logger.info('Creating simple gear scene')
    PROC.create_gear_train(
        engine,
        N=2,
        density=1.0,
        material_name='default'
    )

def create_planetary_gear(engine: TYPES.Engine):
    logger = logging.getLogger("main.setup_scene.create_planetary_gear_scene")
    logger.info('Creating planetary gear scene')
    PROC.create_planetary_gear(
        engine,
        density=1.0,
        material_name='default'
    )
