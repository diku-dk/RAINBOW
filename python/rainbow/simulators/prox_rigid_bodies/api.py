from typing import List, Dict
import rainbow.geometry.surface_mesh as MESH
import rainbow.math.functions as FUNC
import rainbow.geometry.grid3 as GRID
import rainbow.geometry.kdop_bvh as BVH
import rainbow.simulators.prox_rigid_bodies.solver as SOLVER
from rainbow.simulators.prox_rigid_bodies.types import *
import numpy as np


def generate_unique_name(name: str) -> str:
    """
    This function helps to generate unique names, such that one can always locate objects based on name only.

    :param name:   The original name wanted.
    :return:       pre and post appended name string that makes name unique.
    """
    import datetime
    import random

    n = random.random()
    unique_name = name + "_" + str(n) + "_" + str(datetime.datetime.now())
    return unique_name


def create_engine() -> Engine:
    """
    Create Engine.

    :return: A new Engine instance containing the world to be simulated and the actual simulator to use.
    """
    return Engine()


def create_rigid_body(engine, body_name: str) -> None:
    """
    Create rigid body in engine.

    :param engine:     The engine to contain the rigid body.
    :param body_name:  The unique name of the rigid body.
    :return:           Nothing.
    """
    if body_name in engine.bodies:
        raise RuntimeError("connect() rigid body already exist with that name")
    body = RigidBody(body_name)
    body.idx = len(engine.bodies)
    engine.bodies[body_name] = body


def create_mesh(V, T) -> MESH.Mesh:
    """
    Create a mesh instance.
    This function creates a mesh. This is not a simple copy of the
    input arrays, but a spatial reordering to store the mesh more
    efficient in memory.

    :param V:   The vertices of the mesh. Assumed to be N-by-3 float array, where N is number of nodes.
    :param T:   The triangles of the mesh. Assumed to be K-by-3 int array, where K is number of triangles.
    :return:    A mesh instance.
    """
    axis = FUNC.direction_of_most_variance(V)
    mesh = MESH.Mesh(V, T)
    MESH.axis_sort(mesh, axis)
    return mesh


def create_shape(engine, shape_name: str, mesh) -> None:
    """
    This function creates a new shape in the rigid body engine. The shape instance describes
    how a rigid body "looks" like as well as hold information about mass properties, and
    "static" collision detection information.

    Observe, that when a mesh is given as input then the mesh lives in its "model" frame. This
    factory function will transform the mesh and save it in its local "body" frame. However, the
    transformation between model and body frame is kept such that one later can place a rigid body
    in the world using the model space as a reference for the placement and not the body space.

    Much of this is hidden from the common end-user, who just sees her geometries as they are
    created and place these in the world without ever knowing anything about what a body frame
    really is.

    :param engine:           The engine where the new shape should be created in.
    :param shape_name:       A unique name of the new shape.
    :param mesh:             A triangle surface mesh that is supposed to be used to create the geometry of the shape.
    :return:                 Nothing.
    """
    if shape_name in engine.shapes:
        raise RuntimeError(
            "create_shape_from_obj(): shape with that name already exist"
        )
    shape = Shape(shape_name)
    shape.mesh = mesh
    SOLVER.transform_shape_into_body_frame(
        shape
    )  # Note this computes mass and inertia of shape
    max_length = (shape.mesh.V.max(axis=0) - shape.mesh.V.min(axis=0)).max()
    boundary = max(max_length * 0.1, engine.params.envelope * 2)
    shape.grid = GRID.create_signed_distance(
        shape.mesh.V,
        shape.mesh.T,
        engine.params.resolution,
        engine.params.resolution,
        engine.params.resolution,
        boundary,
    )
    engine.shapes[shape_name] = shape


def connect_shape(engine, body_name: str, shape_name: str) -> None:
    """
    This function creates a connection between a given rigid body and a shape.

    A shape can be shared between multiple rigid bodies which is very space efficient. A "ball" simulator would
    ever only need to create a single ball shape, but could have thousands of ball bodies. The simulator would
    only need to store the dynamic changing data but share the static geometric data.

    :param engine:
    :param body_name:
    :param shape_name:
    :return:
    """
    if body_name in engine.bodies:
        body = engine.bodies[body_name]
    else:
        raise RuntimeError("connect() no such rigid body exist with that name")
    if shape_name in engine.shapes:
        shape = engine.shapes[shape_name]
    else:
        raise RuntimeError("connect() no such shape exist with that name")
    body.shape = shape
    # k-DOP bounding volume hierarchies perform their collision tandem traversal in world space. Hence, their
    # bounding volumes live in world space and as bodies move around we constantly need to update the bounding
    # volumes of the BVHs. Therefore, the BVH data structure is stored in the rigid body and not the shape.
    body.bvh = BVH.make_bvh(
        shape.mesh.V,
        shape.mesh.T,
        engine.params.K,
        engine.params.bvh_chunk_size,
        engine.params.envelope,
    )


def create_gravity_force(engine, force_name: str, g: float, up) -> None:
    """
    Create a gravity force type instance in the engine.

    Observe that one can have multiple different gravity forces, and they
    can be acting on different rigid bodies, one can even have multiple
    gravity forces acting on the same rigid body.

    :param engine:      The engine that should store the new gravity force.
    :param force_name:  A unique name for the gravity force that is created.
    :param g:           The gravity acceleration value.
    :param up:          The direction of the gravity acceleration, a unit-vector.
    :return:            Nothing.
    """
    if force_name in engine.forces:
        raise RuntimeError("create_gravity(): Force already exist with that name")
    if g <= 0.0:
        raise RuntimeError(
            "create_gravity(): Illegal value for gravitational acceleration"
        )
    gravity = Gravity(force_name)
    gravity.up = V3.unit(up)
    gravity.g = g
    engine.forces[force_name] = gravity


def create_damping_force(engine, force_name: str, alpha: float, beta: float) -> None:
    """
    Create a linear damping force type instance in the engine.

    Observe that one can have multiple different damping forces, and they
    can be acting on different rigid bodies, one can even have multiple
    damping forces acting on the same rigid body.

    :param engine:      The engine that should store the new gravity force.
    :param force_name:  A unique name for the gravity force that is created.
    :param alpha:       A positive number that indicates the amount of linear velocity damping.
    :param beta:        A positive number that indicates the amount of angular velocity damping.
    :return:            Nothing.
    """
    if force_name in engine.forces:
        raise RuntimeError("create_damping(): Force already exist with that name")
    if alpha <= 0:
        raise RuntimeError("create_damping(): Illegal value for alpha")
    if beta <= 0:
        raise RuntimeError("create_damping(): Illegal value for beta")
    damping = Damping(force_name)
    damping.alpha = alpha
    damping.beta = beta
    engine.forces[force_name] = damping


def connect_force(engine, body_name: str, force_name: str) -> None:
    """
    Connect a force type to a rigid body.

    :param engine:      The engine that contains the rigid body and the force instance.
    :param body_name:   The unique name of the rigid body.
    :param force_name:  The unique name of the force type.
    :return:            Nothing.
    """
    if body_name in engine.bodies:
        body = engine.bodies[body_name]
    else:
        raise RuntimeError("connect_force() no such rigid body exist with that name")
    if force_name in engine.forces:
        force = engine.forces[force_name]
    else:
        raise RuntimeError("connect_force() no such force exist with that name")
    if force in body.forces:
        raise RuntimeError("connect_force() force was already connected to body")
    body.forces.append(force)


def set_position(engine, body_name: str, r, use_model_frame=False) -> None:
    """
    Set the center of mass position of the rigid body.

    :param engine:           The engine that stores the rigid body.
    :param body_name:        The name of the rigid body.
    :param r:                The new position.
    :param use_model_frame:  Boolean flag used to specify if it is the position of the origin of the model frame
                             or the body frame that is placed in the world coordinate system.
    :return:                 Nothing.
    """
    if body_name in engine.bodies:
        body = engine.bodies[body_name]
    else:
        raise RuntimeError("set_position() no such rigid body exist with that name")
    if use_model_frame:

        # r_bf2wcs = body.r # Not needed
        q_bf2wcs = np.copy(body.q)

        r_bf2mf = np.copy(body.shape.r)
        q_bf2mf = np.copy(body.shape.q)
        #                                                                      |q r| |x|
        # By definition we have the rigid body transformations T x = q*x + r = |0 q| |1|
        #
        #  T_bf2wcs =   T_mf2wcs T_bf2mf
        #
        #   |q_bf2wcs  r_bf2wcs|   |q_mf2wcs  r_mf2wcs|  |q_bf2mf   r_bf2mf |
        #   |0            1    | = |0            1    |  |0            1    |
        #
        # So
        #
        #   q_bf2wcs = q_mf2wcs q_bf2mf
        #   r_bf2wcs = q_mf2wcs r_bf2mf + r_mf2wcs      (*)
        #
        # From this we can solve for the current model to world coordinate transformation
        q_mf2wcs = Q.prod(q_bf2wcs, Q.conjugate(q_bf2mf))
        # r_mf2wcs = r_bf2wcs - Q.rotate(q_mf2wcs, r_bf2mf)  # Not needed
        #
        # Now we wish to change the position of model frame origin to (x,y,z) wrt the world frame
        #
        # so
        #
        #   r_mf2wcs = [x, y, z]
        #
        # The orientation of the model frame wrt. world frame is unchanged, hence we want to
        # know what r_bf2wcs should be. For this we use (*) to compute it
        #
        r_mf2wcs = np.copy(r)
        r_bf2wcs = Q.rotate(q_mf2wcs, r_bf2mf) + r_mf2wcs

        # Finally, we can update the origin of the body model frame origin to reflect the desired position of the model
        #  frame origin in the world coordinate system
        body.r = r_bf2wcs
    else:
        body.r = np.copy(r)


def set_orientation(engine, body_name: str, q, use_model_frame=False) -> None:
    """
    Set orientation of rigid body.

    :param engine:           The engine that stores the rigid body.
    :param body_name:        The name of the rigid body.
    :param q:                The new orientation.
    :param use_model_frame:  Boolean flag used to specify if it is the orientation of the origin of the model frame
                             or the body frame that is placed wrt. the world coordinate system.
    :return:                 Nothing.
    """
    if body_name in engine.bodies:
        body = engine.bodies[body_name]
    else:
        raise RuntimeError("set_position() no such rigid body exist with that name")

    if use_model_frame:

        r_bf2wcs = np.copy(body.r)
        q_bf2wcs = np.copy(body.q)

        r_bf2mf = np.copy(body.shape.r)
        q_bf2mf = np.copy(body.shape.q)
        #                                                                      |q r| |x|
        # By definition we have the rigid body transformations T x = q*x + r = |0 q| |1|
        #
        #  T_bf2wcs =   T_mf2wcs T_bf2mf
        #
        #   |q_bf2wcs  r_bf2wcs|   |q_mf2wcs  r_mf2wcs|  |q_bf2mf   r_bf2mf |
        #   |0            1    | = |0            1    |  |0            1    |
        #
        # So
        #
        #   q_bf2wcs = q_mf2wcs q_bf2mf
        #   r_bf2wcs = q_mf2wcs r_bf2mf + r_mf2wcs
        #
        # From this we can solve for the current model to world coordinate transformation
        #
        q_mf2wcs = Q.prod(q_bf2wcs, Q.conjugate(q_bf2mf))
        r_mf2wcs = r_bf2wcs - Q.rotate(q_mf2wcs, r_bf2mf)
        #
        # Now we wish to change the orientation of the model frame origin  wrt the world frame,
        # so we must now have
        #
        #   q_mf2wcs = [qs, qx, qy, qz]
        #
        q_mf2wcs = Q.unit(q)
        #
        # Change the orientation of the model frame means that both the orientation and
        #  position of the body frame will change wrt. the world coordinate system.
        q_bf2wcs = Q.prod(q_mf2wcs, q_bf2mf)
        r_bf2wcs = Q.rotate(q_mf2wcs, r_bf2mf) + r_mf2wcs

        body.q = q_bf2wcs
        body.r = r_bf2wcs
    else:
        body.q = Q.unit(q)


def set_velocity(engine, body_name: str, v) -> None:
    """
    Set the linear velocity of a rigid body.

    :param engine:    The engine that holds the rigid body.
    :param body_name: The name of the rigid body.
    :param v:         The linear velocity that should be assigned to the body.
    :return:          Nothing.
    """
    if body_name in engine.bodies:
        body = engine.bodies[body_name]
    else:
        raise RuntimeError("set_velocity() no such rigid body exist with that name")
    body.v = np.copy(v)


def set_spin(engine, body_name: str, w) -> None:
    """
    Set the angular velocity of a rigid body.

    :param engine:    The engine that holds the rigid body.
    :param body_name: The name of the rigid body.
    :param w:         The angular velocity that should be assigned to the body.
    :return:          Nothing.
    """
    if body_name in engine.bodies:
        body = engine.bodies[body_name]
    else:
        raise RuntimeError("set_spin() no such rigid body exist with that name")
    body.w = np.copy(w)


def set_mass_properties(engine, body_name: str, density: float) -> None:
    """
    Set the mass properties of a given rigid body.

    :param engine:     The engine that contains the rigid body.
    :param body_name:  The name of the rigid body.
    :param density:    The mass density value of the rigid body.
    :return:           Nothing.
    """
    if body_name in engine.bodies:
        body = engine.bodies[body_name]
    else:
        raise RuntimeError(
            "set_mass_properties() no such rigid body exist with that name"
        )
    if body.shape is None:
        raise RuntimeError("set_mass_properties() rigid body did not have a shape")
    body.mass = body.shape.mass * density
    body.inertia = body.shape.inertia * density


def set_body_type(engine, body_name: str, body_type: str) -> None:
    """
    Set the body type of rigid body. It can be either fixed or free.

    :param engine:       The engine where the rigid body is stored.
    :param body_name:    The name of the rigid body.
    :param body_type:    A string indicating the body type 'fixed' or 'free'.
    :return:             Nothing.
    """
    if body_name not in engine.bodies:
        raise RuntimeError("set_body_type() no such rigid body exist with that name")
    body = engine.bodies[body_name]
    if body_type in ["fixed", "Fixed", "FIXED"]:
        body.is_fixed = True
    elif body_type in ["free", "Free", "FREE"]:
        body.is_fixed = False
    else:
        raise RuntimeError("set_body_type(): Unsupported body type found")


def is_fixed_body(engine, body_name: str) -> bool:
    """
    Query if rigid body is fixed.

    :param engine:      The engine that stores the rigid body.
    :param body_name:   The name of the rigid body.
    :return:            Boolean flag telling whether hte rigid body is fixed or nat.
    """
    if body_name not in engine.bodies:
        raise RuntimeError("is_fixed_body() no such rigid body exist with that name")
    body = engine.bodies[body_name]
    return body.is_fixed


def set_body_material(engine, body_name: str, material_name: str) -> None:
    """
    Sets the material of a rigid body.

    The material must have been created as part of a surface interaction before the material
    can be set for a given rigid body.

    :param engine:         The engine that stores the rigid body and material information.
    :param body_name:      The name of the rigid body.
    :param material_name:  The name of the material.
    :return:               Nothing.
    """
    if body_name not in engine.bodies:
        raise RuntimeError(
            "set_body_material() no such rigid body exist with that name"
        )
    body = engine.bodies[body_name]
    if not engine.surfaces_interactions.exist_material(material_name):
        raise RuntimeError("set_body_material() no such material exist")
    body.material = material_name


def create_surfaces_interaction(engine, A: str, B: str, epsilon: float, mu) -> None:
    """
    This function creates a description of the material surface interaction between two named materials.

    :param engine:   The engine that stores information about the surface interaction.
    :param A:        Unique name of first material of interaction.
    :param B:        Unique name of the second material of the interaction.
    :param epsilon:  The value of the coefficient of restitution to use between the two pair of materials.
    :param mu:       The value of the coefficients of friction to use between the material pairs.
    :return:         Nothing.
    """
    if engine.surfaces_interactions.exist_interaction(A, B):
        raise RuntimeError("create_surfaces_interaction() behaviour already exist")
    if epsilon < 0.0:
        raise RuntimeError("create_surfaces_interaction() illegal epsilon value")
    if mu[0] < 0.0:
        raise RuntimeError("create_surfaces_interaction() illegal mu_x value")
    if mu[1] < 0.0:
        raise RuntimeError("create_surfaces_interaction() illegal mu_y value")
    if mu[2] < 0.0:
        raise RuntimeError("create_surfaces_interaction() illegal mu_z value")
    tmp = [A, B]
    tmp.sort()
    key = tuple(tmp)
    behaviour = SurfacesInteraction()
    behaviour.epsilon = epsilon
    behaviour.mu = mu
    engine.surfaces_interactions.storage[key] = behaviour

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
        engine.stepper = SOLVER.SemiImplicitStepper(engine)
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
