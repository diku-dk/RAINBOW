from typing import List, Dict
import rainbow.geometry.grid3 as GRID
import rainbow.geometry.kdop_bvh as BVH
import rainbow.geometry.spatial_hashing as HASH_GRID
import rainbow.math.functions as FUNC
import rainbow.math.vector3 as V3
import rainbow.geometry.surface_mesh as SURF_MESH
import rainbow.geometry.volume_mesh as MESH
import rainbow.simulators.prox_soft_bodies.solver as SOLVER
from rainbow.simulators.prox_soft_bodies.types import *

import numpy as np


def create_engine() -> Engine:
    """
    Create Engine.

    :return: A new Engine instance containing the world to be simulated and the actual simulator to use.
    """
    return Engine()


def create_soft_body(engine, body_name, V, T) -> None:
    """
    Create a soft body in the given engine.

    :param engine:      A reference to the engine where the soft body should be created.
    :param body_name:   The name of the new body.
    :param V:           A N-by-3 array of mesh vertex coordinates  N is number of vertices.
    :param T:           A K-by-4 array of mesh tetrahedral elements. K is number of tetrahedrons.
    :return:            Nothing.
    """
    if body_name in engine.bodies:
        raise RuntimeError("Soft body already exist with that name")

    # Create body
    body = SoftBody(body_name)
    body.idx = len(engine.bodies)
    engine.bodies[body_name] = body

    # Improve spatial/memory layout of volume mesh
    axis = FUNC.direction_of_most_variance(V)
    mesh = MESH.T4Mesh(V, T)
    SURF_MESH.axis_sort(mesh, axis)

    # Setup volume mesh related data and do all material space pre-computations
    body.T = np.array(mesh.T, copy=True, dtype=np.int32)
    body.x0 = np.array(mesh.V, copy=True, dtype=np.float64)
    body.vol0 = MESH.compute_volumes(body.x0, body.T)
    body.gradN0 = SOLVER.Native.compute_outward_face_vectors(body.x0, body.T)
    body.invD0 = SOLVER.Native.compute_inverse_D0(body.x0, body.T)

    # Next we extract the triangle surface mesh from the volume mesh. This is used for visualization.
    body.neighbors = MESH.compute_neighbors(body.T)
    body.surface, body.owners = MESH.compute_surface(body.T, body.neighbors)

    # Setup material space signed distance field
    max_length = (body.x0.max(axis=0) - body.x0.min(axis=0)).max()
    boundary = max(max_length * 0.1, engine.params.envelope * 2)
    body.grid = GRID.create_signed_distance(
        body.x0,
        body.surface,
        engine.params.resolution,
        engine.params.resolution,
        engine.params.resolution,
        boundary,
    )

    # Create spatial volume mesh variables
    body.x = np.array(mesh.V, copy=True, dtype=np.float64)
    body.u = np.zeros(V.shape, dtype=np.float64)

    
    # If we use spatial hashing we need to setup the hash grid, otherwise we create a BVH
    if engine.params.use_spatial_hashing:
        engine.hash_grid.increment_hash_table_size(len(body.surface))
    
        if engine.hash_grid.cell_size == 0:
            engine.hash_grid.cell_size = HASH_GRID.HashGird.compute_optial_cell_size(body.x0, body.surface)
        else :
            engine.hash_grid.cell_size = (HASH_GRID.HashGird.compute_optial_cell_size(body.x0, body.surface)+engine.hash_grid.cell_size)/2 
    else:
        # Create bounding volume hierarchy data-structure (BVH), this will always be updated to live in
        # spatial coordinates and is tested against the signed distance field (who lives in constant material space) to
        # generate contact points.
        body.bvh = BVH.make_bvh(
            body.x,
            body.surface,
            engine.params.K,
            engine.params.bvh_chunk_size,
            engine.params.envelope,
        )

    # To have proper global indexing into assembled matrices and vectors we need to know this body nodel
    # index offset into this global space.
    body.offset = engine.number_of_nodes
    engine.number_of_nodes += len(body.x0)

    

def create_dirichlet_conditions(engine, body_name, phi) -> None:
    """
    This function creates Dirichlet conditions on the given body
    for all nodes that in material coordinate are "inside" the given phi-function.
    Inside just mean that phi returns a non-positive number.

    :param engine:       A reference to the engine where the body comes from.
    :param body_name:    The name of the body that we want to add conditions to.
    :param phi:          An implicit function that maps from R^3 to R. Negative
                         values are inside, positive values outside.
    :return:             Nothing.
    """
    if body_name not in engine.bodies:
        raise ValueError("Engine does not contain soft body of name", body_name)

    body = engine.bodies[body_name]
    for i in range(len(body.x0)):
        if phi(body.x0[i]) <= 0:
            body.dirichlet_conditions.append(DirichletCondition(i, body.x0[i]))


def create_traction_conditions(engine, body_name, phi, load) -> None:
    """
    Create traction conditions on all triangular surface faces of the mesh that
    are inside the given phi function. Inside test of a triangle means all vertices
    of the triangle are inside phi.

    :param engine:       A reference to the engine where the body comes from.
    :param body_name:    The name of the body that we want to add conditions to.
    :param phi:          An implicit function that maps from R^3 to R. Negative
                         values are inside, positive values outside.
    :param load:         A 3D vector that holds the load value to be applied.
    :return:             Nothing.
    """
    if not isinstance(load, np.ndarray):
        raise TypeError
    if body_name not in engine.bodies:
        raise ValueError("Engine does not contain soft body of name", body_name)
    body = engine.bodies[body_name]
    for triangle in body.surface:
        # Check that the triangle is inside phi volume
        i = triangle[0]
        j = triangle[1]
        k = triangle[2]
        if phi(body.x0[i]) <= 0 and phi(body.x0[j]) <= 0 and phi(body.x0[k]) <= 0:
            # Triangle was inside, so we create a traction condition for it.
            body.traction_conditions.append(TractionCondition(i, j, k, load))


def set_velocity(engine, body_name, v) -> None:
    """
    Set the global velocity of a given soft body.

    :param engine:      A reference to the engine that has the body.
    :param body_name:   The name of the body.
    :param v:           The velocity value to be set for all nodes in the mesh.
    :return:            Nothing.
    """
    if len(v) != 3:
        raise ValueError("v input must be of length 3", body_name)
    if body_name not in engine.bodies:
        raise ValueError("Engine does not contain soft body of name", body_name)
    body = engine.bodies[body_name]
    for i in range(len(body.u)):
        body.u[i] = v


def get_material_coordinates(engine, body_name) -> np.ndarray:
    """
    Get material (undeformed) coordinates of the given body.
    This interface function returns a copy of the coordinates.

    :param engine:      A reference to the engine that has the body.
    :param body_name:   The name of the body.
    :return:            The material (undeformed) coordinates.
    """
    if body_name not in engine.bodies:
        raise ValueError("Engine does not contain soft body of name", body_name)
    body = engine.bodies[body_name]
    x0 = np.copy(body.x0)
    return x0


def set_spatial_coordinates(engine, body_name, x) -> None:
    """
    Set spatial (deformed) coordinates of given body.
    This interface function sets a copy of the provided input data.

    :param engine:      A reference to the engine that has the body.
    :param body_name:   The name of the body.
    :param x:           The new deformed coordinates.
    :return:            Nothing.
    """
    if body_name not in engine.bodies:
        raise ValueError("Engine does not contain soft body of name", body_name)
    body = engine.bodies[body_name]
    if x.shape != body.x.shape:
        raise ValueError("x input must have shape ", body.x.shape)
    body.x = np.copy(x)


def set_type(engine, body_name, body_type) -> None:
    """
    Toggle if a soft body is fixed (can not move at all) or free to move.

    :param engine:      A reference to the engine that has the body.
    :param body_name:   The name of the body.
    :param body_type:   The type of body, can be "fixed" or "free" (character case does not matter).
    :return:            Nothing.
    """
    if body_name not in engine.bodies:
        raise ValueError("Engine does not contain soft body of name", body_name)
    if not body_type.lower() in ["fixed", "free"]:
        raise ValueError("Body type must be fixed or free", body_type)
    body = engine.bodies[body_name]
    body.is_fixed = False
    if body_type.lower() == "fixed":
        body.is_fixed = True


def create_surfaces_interaction(engine, material_A, material_B, mu) -> None:
    """
    Set parameters that describe the interaction between the given materials.

    :param engine:      The engine that should store the interaction.
    :param material_A:  The name of material A.
    :param material_B:  The name of material B.
    :param mu:          The value of the coefficient of friction between the two materials given.
    :return:            Nothing.
    """
    # 2022-05-05 Kenny TODO add test that mu is a non-negative real number.
    if material_A not in engine.materials:
        raise ValueError("Engine did not have material with name ", material_A)
    if material_B not in engine.materials:
        raise ValueError("Engine did not have material with name ", material_B)
    if engine.surfaces_interactions.exist_interaction(material_A, material_B):
        raise RuntimeError("Surfaces interaction  already exist")
    if (np.array(mu) < 0.0).any():
        raise RuntimeError("Illegal mu value")
    key = (
        (material_A, material_B)
        if material_A < material_B
        else (material_B, material_A)
    )
    interaction = SurfacesInteraction()
    interaction.mu[0] = mu
    engine.surfaces_interactions.storage[key] = interaction


def create_material(engine, material_name) -> None:
    """
    Creates a new material with given name inside the engine.

    :param engine: b         The engine that holds the new material.
    :param material_name:    The name of the material.
    :return:                 Nothing.
    """
    if material_name in engine.materials:
        raise ValueError("Engine already have a material with name ", material_name)
    material = MaterialDescription()
    material.name = material_name
    engine.materials[material_name] = material


def set_material(engine, body_name, material_name) -> None:
    """
    Assign the given material to the given body.

    :param engine:         The engine that contains the body and the material.
    :param body_name:      The name of the body.
    :param material_name:  The name of the material.
    :return:               Nothing.
    """
    if body_name not in engine.bodies:
        raise ValueError("Engine does not contain soft body of name", body_name)
    body = engine.bodies[body_name]
    if material_name not in engine.materials:
        raise ValueError("Engine did not have a material with name ", material_name)
    material = engine.materials[material_name]
    body.material_description = material
    body.M_array = SOLVER.Native.compute_mass_element_array(
        body.material_description.rho,
        body.vol0,
        body.T,
        body.is_lumped
    )
    body.C_array = SOLVER.Native.compute_damping_element_array(
        body.material_description.c, body.vol0, body.T, body.is_lumped
    )


def set_elasticity(engine, material_name, E, nu) -> None:
    """
    Set elasticity parameters of given material.

    :param engine:         The engine that holds the material.
    :param material_name:  The name of the material.
    :param E:              The value of Young modulus.
    :param nu:             The value of Poisson ratio.
    :return:               Nothing.
    """
    if material_name not in engine.materials:
        raise ValueError("Engine did not have a material with name ", material_name)
    if E <= 0:
        raise ValueError("E must be positive")
    if nu < 0 or nu > 0.5:
        raise ValueError("nu must be in range 0 to 1/2")
    material = engine.materials[material_name]
    material.E = E
    material.nu = nu


def set_viscosity(engine, material_name, c) -> None:
    """
    Set viscosity parameter of given material.

    :param engine:          The engine holding the material.
    :param material_name:   The name of the material.
    :param c:               The value of the viscosity parameter.
    :return:                Nothing.
    """
    if material_name not in engine.materials:
        raise ValueError("Engine did not have a material with name ", material_name)
    if c < 0:
        raise ValueError("c must be non-negative")
    material = engine.materials[material_name]
    material.c = c


def set_mass_density(engine, material_name, rho) -> None:
    """
    Set mass density of given material.

    :param engine:          The engine holding the material.
    :param material_name:   The name of the material.
    :param rho:             The mass density value.
    :return:                Nothing.
    """
    if material_name not in engine.materials:
        raise ValueError("Engine did not have a material with name ", material_name)
    if rho < 0:
        raise ValueError("rho must be non-negative")
    material = engine.materials[material_name]
    material.rho = rho


def set_constitutive_model(engine, material_name, model) -> None:
    """
    Set the constitutive model of a given material.

    :param engine:          The engine holding the material.
    :param material_name:   The name of the material.
    :param model:           A reference to a constitutive model class.
    :return:                Nothing.
    """
    if not hasattr(model, "pk1_stress"):
        raise TypeError
    if not hasattr(model, "energy_density"):
        raise TypeError
    if material_name not in engine.materials:
        raise ValueError("Engine did not have a material with name ", material_name)
    material = engine.materials[material_name]
    material.constitutive_model = model


def set_gravity(engine, body_name, g) -> None:
    """
    Set the gravity acceleration vector acting on the given body.

    :param engine:         The engine with the body.
    :param body_name:      The name of the body.
    :param g:              The gravity acceleration vector.
    :return:               Nothing.
    """
    # 2022-03-27 Kenny TODO: We currently allow a per-body gravity force to be specified. One could define
    #                   gravity to be the same in the whole world instead. In this case gravity could be
    #                   stored in the engine class.
    if body_name not in engine.bodies:
        raise ValueError("Engine does not contain soft body of name", body_name)
    if len(g) != 3:
        raise ValueError("Gravity acceleration must be a 3D vector")
    body = engine.bodies[body_name]
    body.gravity = V3.make(g[0], g[1], g[2])


def simulate(engine, T: float, debug_on: bool = False) -> None:
    """
    Simulate forward in time.

    :param engine:    The engine holding the world to be simulated.
    :param T:         The time to simulate forward.
    :param debug_on:  Boolean flag indicating if debug info should be generated or not.
    :return:          None
    """
    if T <= 0:
        raise ValueError("Time must be positive")
    if engine.stepper is None:
        engine.stepper = SOLVER.SemiImplicitStepper(engine, debug_on)
    T_left = T
    while T_left:
        dt = min(T_left, engine.params.time_step)
        engine.stepper.step(dt, engine, debug_on)
        T_left -= dt


def get_log(engine: Engine) -> List[Dict]:
    """
    Retrive log with debug information and timings etc.

    :param engine:  The engine to retrieve the log from.
    :return:        The log is basically a list of dictionaries. One dictionary for each invocation of the stepper.
    """
    return engine.stepper.log


# 2022-03-27 Kenny TODO: End users might want to be able to read back the contact force solutions too. This could
#                   be both as nodal forces on the surface mesh, or as normal load and shear stress on the surface
#                   mesh.

# 2022-03-27 Kenny TODO: Would be nice to add capabilities for exporting to VTK files, so users can post-process
#                    data in paraview.
