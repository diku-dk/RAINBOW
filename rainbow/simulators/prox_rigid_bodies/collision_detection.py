import rainbow.geometry.grid3 as GRID
import rainbow.geometry.kdop_bvh as BVH
from rainbow.simulators.prox_rigid_bodies.types import *
from rainbow.util.timer import Timer

from itertools import combinations
import numpy as np


def _update_bvh(engine: Engine, profile_data: dict, profiling_on: bool):
    """
    This function updates the bounding volume hierarchies of the rigid bodies to reflect the current
    world space geometry of the bodies. This is necessary to make sure that we later in the narrow phase
    collision detection function will be working on the geometry where it is actually correctly placed
    in the world.

    :param engine:           The current engine instance we are working with.
    :param profile_data:     A dictionary where to add more profiling and timing measurements.
    :param profiling_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:                 None.
    """
    update_bvh_timer = None
    if profiling_on:
        update_bvh_timer = Timer("update_bvh")
        update_bvh_timer.start()
    for body in engine.bodies.values():
        # Bounding volume hierarchy (BVH) traversal testing is done
        # in world space. Hence, before we refit the bounding volumes (BVs) of
        # the BVH trees we need to update the "geometry" that the BVH bounds to
        # reflect its current world space position. For rigid bodies this is simply
        # done by applying a rigid body transform to all the nodes of the triangle mesh.
        V_world = Q.rotate_array(body.q, body.shape.mesh.V) + body.r
        # After we have updated the triangle mesh nodes we can invoke the refit
        # sub-routine from the BVH module.
        BVH.refit_bvh(
            V_world,
            body.shape.mesh.T,
            body.bvh,
            engine.params.K,
            engine.params.envelope,
        )
    if profiling_on:
        update_bvh_timer.end()
        profile_data["update_bvh"] = update_bvh_timer.elapsed


def _narrow_phase(engine: Engine, profile_data: dict, profiling_on: bool) -> dict:
    """
    This function performs narrow phase collision detection between overlapping bodies in the engine. Narrow
    phase collision detection means looking more closely for potential overlapping triangle pairs between
    a pair of bodies.

    :param engine:      The current engine instance we are working with.
    :param profile_data:     A dictionary where to add more profiling and timing measurements.
    :param profiling_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary where keys are overlapping bodies and values are a list of potential colliding triangle pairs.
    """
    narrow_phase_timer = None
    if profiling_on:
        narrow_phase_timer = Timer("narrow_phase", 8)
        narrow_phase_timer.start()
    # We do not currently have any intelligent broad phase collision detection system to identify a pair
    # of sufficiently close bodies. Hence, we simply just test all possible pair of bodies for collision. This
    # is done by invoking a tandem traversal on the BVH trees of the two bodies. The tandem traversal descent
    # down through the BVH trees whenever it finds overlapping volumes. In the end when it encounters two
    # leaf bounding volumes it reports the two enclosing triangles (one from each body) as a potential
    # pair-wise triangle overlap. The output from the tandem traversal is a list of triangle pairs. Each
    # triangle pair is two index values that are "pointing" to the respective triangles.
    overlaps = {}
    for bodyA, bodyB in combinations(engine.bodies.values(), r=2):
        results = BVH.traversal(bodyA.bvh, bodyB.bvh)
        # It may be that the body pairs are not overlapping at all and in this case the
        # triangle pair list will be empty. We do not wish to through body overlap pairs further
        # down our collision detection pipeline, so we test if there are any triangle pairs to
        # process or not. If there are no triangles we do not report anything for the body pairs.
        if len(results) > 0:
            overlaps[(bodyA, bodyB)] = np.array(results, dtype=np.int32)
    if profiling_on:
        narrow_phase_timer.end()
        profile_data["narrow_phase"] = narrow_phase_timer.elapsed
        profile_data["number_of_overlaps"] = np.sum(
            [len(results) for results in overlaps.values()]
        )
    return overlaps


def _compute_contacts(engine: Engine, bodyA: RigidBody, bodyB: RigidBody, trianglesA, profile_data: dict, profiling_on: bool) -> None:
    """
    This function computes contacts between triangles from body A against the signed distance field of body B.

    :param engine:           The current engine instance we are working with.
    :param bodyA:            Reference to body A.
    :param bodyB:            Reference to body B.
    :param trianglesA:       Array of triangles from body A that may be colliding with body B.
    :param profile_data:     A dictionary where to add more profiling and timing measurements.
    :param profiling_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:                 Nothing.
    """
    model_space_update_timer = None
    contact_optimization_timer = None
    contact_point_generation_timer = None
    if profiling_on:
        model_space_update_timer = Timer("model_space_update")
        contact_optimization_timer = Timer("contact_optimization")
        contact_point_generation_timer = Timer("contact_point_generation")
        model_space_update_timer.start()

    # TODO 2021-12-31 Kenny: The parts 'bodyA.shape.mesh' are very specific for the rigid body type. This
    #  makes it hard to reuse the code across different simulators. It would be nice with a more generic design
    #  of the collision detection module, so it could be separated out into an individual shared component
    #  between multiple simulators.

    # The whole contact point computation happens in the local space of the signed distance filed (SDF) of
    # body B. Hence, we must transform information that lives in body A's local body space into the body space of
    # body B. This means we will get A's vertices in B's local body frame.
    # TODO 2022-01-02 Kenny: We currently transform all vertices of A. This makes pretty and simple code, but
    #  if body A has a large mesh and only a few triangles of body A are in actual contact then we are burning
    #  a lot of flops for nothing. The coordinate transformation could be done locally for each triangle instead
    #  to save computations.
    V_w = Q.rotate_array(bodyA.q, bodyA.shape.mesh.V) + bodyA.r
    V_b = Q.rotate_array(Q.conjugate(bodyB.q), V_w - bodyB.r)
    if profiling_on:
        model_space_update_timer.end()
    # Loop over all triangles from body A that are colliding with body B
    for t_a in trianglesA:
        if profiling_on:
            contact_optimization_timer.start()
        # We perform a Frank-Wolfe optimization algorithm, https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
        # This was shown to be very efficient in this paper: https://dl.acm.org/doi/10.1145/3384538
        #
        # First we initialize our optimization method for searching for deepest penetrating
        # point on the triangle from body A. This is done by first extracting the three
        # corner vertices of body A's triangle
        p_bs = V_b[bodyA.shape.mesh.T[t_a], :]
        # then we compute the norm of the gradients at corner points of the triangle
        gradients = np.linalg.norm(
            [GRID.get_gradient(bodyB.shape.grid, p_b) for p_b in p_bs], axis=1
        )
        # and lastly we pick the corner point with the smallest gradient norm
        x_i = p_bs[np.argmin(gradients)]

        # Next we are ready for performing the actual optimization, and we set up
        # a loop running for a maximum allowed number of iterations.
        for i in range(engine.params.contact_optimization_max_iterations):
            # TODO 2022-01-02 Kenny: The computation of the objectives repeats the same gradient computation three
            #  times. This could be optimized by simply storing the value of the norm of the gradient
            #  of x_i in a local variable.
            # Pick the triangle vertex 's_i' which minimizes the dot product with
            # the current gradient at 'x_i'. That is the vertex with "largest" descent.
            objectives = [
                np.dot(s_i, GRID.get_gradient(bodyB.shape.grid, x_i)) for s_i in p_bs
            ]
            vertex = np.argmin(objectives)
            s_i = p_bs[vertex]
            # Knowing that 's_i' has a "better" descent direction we update 'x_i' by "dragging" it
            # in the direction of 's_i'. The step-size that we update 'x_i' with is given by alpha.
            # We decrease the value of alpha as we iterate to ensure we do not overstep our minimizer.
            alpha = 2 / (i + 2)
            x_i = x_i + alpha * (s_i - x_i)
            # Before continuing to the next iterate we check for convergence to see if we can make
            # an early exit. We use a simple tolerance test. Note that our "objective" is more like the directional
            # derivative. Hence, if the smallest directional derivative gets slightly positive then it means we will
            # move away from the "minimizer" and we can not find any other corner point with a better descent direction.
            if objectives[vertex] > engine.params.contact_optimization_tolerance:
                break
        if profiling_on:
            contact_optimization_timer.end()
            contact_point_generation_timer.start()
        # We have now optimized for the deepest penetrating point on the triangle, and now we can use this
        # point and the SDF from body B to generate a contact point between body A and body B. First we test
        # if our triangle point is even inside axis aligned bounding box (AABB) around the SDF of body B.
        if GRID.is_inside(bodyB.shape.grid, x_i):
            # If we are inside the AABB then we look up the actual SDF value at the point. If the SDF value
            # is non-positive then we know we have a "true" contact point.
            phi = GRID.get_value(bodyB.shape.grid, x_i)
            if phi <= engine.params.envelope:
                # In which case we can generate the contact point information. We use the SDF value as an
                # estimate of the penetration depth, and we take the gradient from the SDF at the contact
                # point as the normal value. However, we must remember that the whole optimization problem
                # was solved in the local body frame of body B. Hence, we must warp back the
                # contact point information into world space before we can report back the new contact point.
                gap = phi
                n_b = GRID.get_gradient(bodyB.shape.grid, x_i)
                n_w = Q.rotate(bodyB.q, n_b)
                if V3.norm(n_w) > 0:
                    p_w = Q.rotate(bodyB.q, x_i) + bodyB.r
                    cp = ContactPoint(bodyB, bodyA, p_w, V3.unit(n_w), gap)
                    engine.contact_points.append(cp)
        if profiling_on:
            contact_point_generation_timer.end()
    # Before we exit we just make sure we collect any stats and timings.
    if profiling_on:
        if "model_space_update" not in profile_data:
            profile_data["model_space_update"] = 0
        profile_data["model_space_update"] += model_space_update_timer.total
        if "contact_optimization" not in profile_data:
            profile_data["contact_optimization"] = 0
        profile_data["contact_optimization"] += contact_optimization_timer.total
        if "contact_point_generation" not in profile_data:
            profile_data["contact_point_generation"] = 0
        profile_data["contact_point_generation"] += contact_point_generation_timer.total


def _contact_determination(overlaps, engine: Engine, profile_data: dict, profiling_on: bool) -> None:
    """
    This function performs determination of contact points between all pairs of overlapping bodies. The function
     essentially post-process the overlap-data that was computed by the narrow-phase collision detection function.

    :param overlaps:    A dictionary where keys are overlapping bodies and values are a list of potential
                        colliding triangle pairs.
    :param engine:      The current engine instance we are working with.
    :param profile_data:     A dictionary where to add more profiling and timing measurements.
    :param profiling_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:                 None
    """
    contact_determination_timer = None
    if profiling_on:
        contact_determination_timer = Timer("contact_determination", 8)
        contact_determination_timer.start()
    engine.contact_points = []
    for key, results in overlaps.items():
        # TODO 2021-12-31 Kenny: As we only test triangles against SDF
        #  and BVH returns overlapping triangle pairs and since a triangle
        #  from one body can easily overlap with many triangles from the
        #  other body we have to reduce the pair-wise info to unique list
        #  of triangles that could be overlapping with the other body. It
        #  is an overhead we have to pay due to how the BVH traversal
        #  algorithm works. A different algorithm than the current BVH
        #  traversal may not need this overhead.
        _compute_contacts(
            engine,
            key[0],  # Body A
            key[1],  # Body B
            np.unique(
                results[:, 0]
            ),  # 1st column will be all triangles from A that may collide with B
            profile_data,
            profiling_on,
        )
        _compute_contacts(
            engine,
            key[1],  # Body B
            key[0],  # Body A
            np.unique(
                results[:, 1]
            ),  # 2nd column will be all triangles from B that may collide with A
            profile_data,
            profiling_on,
        )
    if profiling_on:
        contact_determination_timer.end()
        profile_data["contact_determination"] = contact_determination_timer.elapsed


def _contact_reduction(engine: Engine, profile_data: dict, profiling_on: bool) -> None:
    """
    During contact point computation it may happen that different colliding triangles of one body results
    in the same contact point locations wrt to the other signed distance field of the other body. Imagine a spiky
    polygonal cone pointing into a spherical shape. Here all triangles of the spike will result in the same
    contact point at the spiky tip. If the spiky polygonal cone has N triangle faces on the spike then we
     will have N redundant contact points. This redundancy is usually bad for other numerical sub-routines for
    computing impacts or contact forces. Hence, the purpose of this step is to eliminate redundant
    contact point information. One may think of this as a kind of post-filtering process to clean up the
     contact point information.

    :param engine:           The current engine instance we are working with.
    :param profile_data:     A dictionary where to add more profiling and timing measurements.
    :param profiling_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:                 None.
    """
    reduction_timer = None
    if profiling_on:
        reduction_timer = Timer("contact_point_reduction", 8)
        reduction_timer.start()
    # TODO 2020-09-07 Kristian: This brute force implementation can be implemented better
    reduced_list = []
    for cp1 in engine.contact_points:
        unique = True
        for cp2 in reduced_list:
            if {cp1.bodyA, cp1.bodyB} == {cp2.bodyA, cp2.bodyB} and (
                cp1.p == cp2.p
            ).all():
                unique = False
        if unique:
            reduced_list.append(cp1)
    engine.contact_points = reduced_list
    if profiling_on:
        reduction_timer.end()
        profile_data["contact_point_reduction"] = reduction_timer.elapsed


def run_collision_detection(engine: Engine, profile_data: dict, profiling_on: bool) -> None:
    """
    This function invokes the whole collision detection pipeline on all the bodies
    currently active in the provided engine instance.

    It is assumed that state information of all bodies have been correctly updated to reflect the
    instant-of-time position where one wish to perform the collision detection at.

    :param engine:           The current engine instance we are working with.
    :param profile_data:     A dictionary where to add more profiling and timing measurements.
    :param profiling_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:                 None.
    """
    collision_detection_timer = None
    if profiling_on:
        collision_detection_timer = Timer("collision_detection")
        collision_detection_timer.start()

    _update_bvh(engine, profile_data, profiling_on)
    overlaps = _narrow_phase(engine, profile_data, profiling_on)
    _contact_determination(overlaps, engine, profile_data, profiling_on)
    _contact_reduction(engine, profile_data, profiling_on)

    if profiling_on:
        collision_detection_timer.end()
        profile_data["collision_detection_time"] = collision_detection_timer.elapsed
