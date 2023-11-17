import rainbow.geometry.grid3 as GRID
import rainbow.geometry.kdop_bvh as BVH
import rainbow.geometry.barycentric as BC
from rainbow.geometry.aabb import AABB
from rainbow.simulators.prox_soft_bodies.types import *
from rainbow.util.timer import Timer
import numpy as np
from itertools import combinations, product
from collections import defaultdict


def _update_bvh(engine, stats, debug_on):
    """
    This function updates the bounding volume hierarchies of the soft bodies to reflect the current
    world space geometry of the bodies. This is necessary to make sure that we later in the narrow phase
    collision detection function will be working on the geometry where it is actually correctly placed
    in the world.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary with profiling and timing measurements.
    """
    update_bvh_timer = None
    if debug_on:
        update_bvh_timer = Timer("update_bvh")
        update_bvh_timer.start()
    for body in engine.bodies.values():
        # Bounding volume hierarchy (BVH) traversal testing is done
        # in world space. Hence, before we refit the bounding volumes (BVs) of
        # the BVH trees we need to update the "geometry" that the BVH bounds to
        # reflect its current world space position. That is we used the current
        # spatial coordinates of the soft bodies to refit the BVs in the BVH tree.
        BVH.refit_bvh(
            body.x, body.surface, body.bvh, engine.params.K, engine.params.envelope
        )
    if debug_on:
        update_bvh_timer.end()
        stats["update_bvh"] = update_bvh_timer.elapsed
    return stats


def _is_share_vertex(tri1, tri2):
    """ Test if two triangles of a same body share a vertex.

    Args:
        tri1 (ArrayLike): coordinates of the first triangle
        tri2 (ArrayLike): coordinates of the second triangle
    
    Returns:
        bool: True if the two triangles share a vertex, False otherwise
    """
    return len(np.intersect1d(np.array(tri1), np.array(tri2))) > 0


def _triangle_intersection(tri1, tri2):
    """ Test if two triangles of a same body intersect
        To achieve performance, this function is adapted from Moller-Trumbore intersection algorithm,
        which is described in https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm.
        The idea is to check if the intersection point of the two triangles is inside both triangles.
        Steps:
        1. Check if the two triangles are parallel or not, if they are parallel, return False
        2. Check if the intersection point is inside the first triangle, if not, return False
        3. Check if the intersection point is inside the second triangle, if inside, return True, otherwise return False
        
    Args:
        tri1 (ArrayLike): coordinates of the first triangle
        tri2 (ArrayLike): coordinates of the second triangle

    Returns:
        bool: True if the two triangles intersect, False otherwise
    """
    v1, v2, v3 = tri1
    u1, u2, u3 = tri2

    e1 = v2 - v1
    e2 = v3 - v1
    normal_tri2 = np.cross(u2 - u1, u3 - u1)
    a = np.dot(e1, normal_tri2)

    # Check if the two triangles are parallel or not
    if a > -np.finfo(float).eps and a < np.finfo(float).eps:
        return False 

    # Check the intersection point is inside the triangle(tri1)
    f = 1.0/a
    s = u1 - v1
    u = f * np.dot(s, normal_tri2)
    if u < 0.0 or u > 1.0:
        return False

    q = np.cross(s, e1)
    v = f * np.dot(u2 - u1, q)
    if v < 0.0 or u + v > 1.0:
        return False

    # Check the intersection point is also inside the other triangle(tri2)
    t = f * np.dot(e2, q)
    if t > np.finfo(float).eps:
        return True

    return False


def _is_self_collision(tri1, tri2, tri1_aabb, tri2_aabb):
    """ Check if two triangles of a same body are self-colliding

    Args:
        tri1 (ArrayLike): coordinates of the first triangle
        tri2 (ArrayLike): coordinates of the second triangle
        tri1_aabb (AABB): the AABB of the first triangle
        tri2_aabb (AABB): the AABB of the second triangle

    Returns:
        bool: True if the two triangles are self-colliding, False otherwise
    """
    return (not _is_share_vertex(tri1, tri2) and
            AABB.is_overlap(tri1_aabb, tri2_aabb) and
            _triangle_intersection(tri1, tri2))


def _spatial_hashing_narrow_phase(engine, stats, debug_on):
    """ Use spatial hashing to find the overlapping triangles

    Args:
        engine (Engine): The current engine instance we are working with.
        stats (dict): A dictionary where to add more profiling and timing measurements.
        debug_on (bool): Boolean flag for toggling debug (aka profiling) info on and off.

    Returns:
        (List, dict):  A tuple with body pair overlap information and a dictionary with profiling and
                        timing measurements.
    """
    narrow_phase_timer = None
    if debug_on:
        narrow_phase_timer = Timer("narrow_phase", 8)
        narrow_phase_timer.start()
    
    cell_size = engine.hash_grid.cell_size
    if cell_size <= 0.0:
        raise ValueError("Cell size must be greater than zero")
    
    time_stamp = engine.params.time_stamp
    results = defaultdict(set)

    for body in engine.bodies.values():
        tri_vertices = body.x[body.surface, :]
        # Compute the AABB of each triangle by vectorizing the min/max operation
        tri_aabb_min = np.min(tri_vertices, axis=1)
        tri_aabb_max = np.max(tri_vertices, axis=1)
        cell_min = (tri_aabb_min / cell_size).astype(int)
        cell_max = (tri_aabb_max / cell_size).astype(int) + 1

        # Traverse the cells in the AABB of each triangle and insert the triangle into the hash table
        for tri_idx, (c_min, c_max) in enumerate(zip(cell_min, cell_max)):
            cell_ranges = [range(cmi, cma) for cmi, cma in zip(c_min, c_max)]
            for i, j, k in product(*cell_ranges):
                tri_aabb = AABB(tri_aabb_min[tri_idx], tri_aabb_max[tri_idx])
                overlaps = engine.hash_grid.insert(i, j, k, tri_idx, body.idx, 
                                                   tri_aabb, 
                                                   time_stamp)
                # Check all triangles in the cell to see if they overlap with the current triangle
                if len(overlaps) > 0:
                    for overlap_tri_idx, overlap_body_idx, overlap_tri_aabb  in overlaps:
                        overlap_body = list(engine.bodies.values())[overlap_body_idx]
                        if overlap_body_idx == body.idx:
                            # Potential self-collision
                            if _is_self_collision(body.x[body.surface[tri_idx]], overlap_body.x[overlap_body.surface[overlap_tri_idx]], tri_aabb, overlap_tri_aabb):
                                results[(body, overlap_body)].add((tri_idx, overlap_tri_idx))
                        else:
                            # Potential collision with another body
                            if AABB.is_overlap(tri_aabb, overlap_tri_aabb):
                                results[(body, overlap_body)].add((tri_idx, overlap_tri_idx))

    if debug_on:
        narrow_phase_timer.end()
        stats["narrow_phase"] = narrow_phase_timer.elapsed
        stats["number_of_overlaps"] = np.sum(
            [len(result) for result in results.values()]
        )

    results = {key: np.array(list(value), dtype=np.int32) for key, value in results.items()}

    return results, stats


def _narrow_phase(engine, stats, debug_on):
    """

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A tuple with body pair overlap information and a dictionary with profiling and
                        timing measurements.
    """
    narrow_phase_timer = None
    if debug_on:
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
    if debug_on:
        narrow_phase_timer.end()
        stats["narrow_phase"] = narrow_phase_timer.elapsed
        stats["number_of_overlaps"] = np.sum(
            [len(results) for results in overlaps.values()]
        )
    return overlaps, stats


def _xform_triangle_to_model_space(P, X, X0):
    """
    Converts a world space triangle into the material space of a tetrahedron.

    :param P:   The input triangle points in world space
    :param X:   The input tetrahedron corners in world space.
    :param X0:  The input tetrahedron corners in material space.
    :return:    The return values are the triangle corner points in tetrahedron B's material coordinate space.
    """
    assert P.shape == (3, 3)
    assert X.shape == (4, 3)
    assert X0.shape == (4, 3)
    P0 = np.zeros(P.shape, dtype=np.float64)
    w0 = BC.compute_barycentric_tetrahedron(X[0], X[1], X[2], X[3], P[0])
    w1 = BC.compute_barycentric_tetrahedron(X[0], X[1], X[2], X[3], P[1])
    w2 = BC.compute_barycentric_tetrahedron(X[0], X[1], X[2], X[3], P[2])
    P0[0] = np.dot(X0.T, w0)
    P0[1] = np.dot(X0.T, w1)
    P0[2] = np.dot(X0.T, w2)
    return P0


def _xform_contact_to_world(p, n, XA, XB, X0B):
    """
    Convert contact point and normal from model space of body B into world coordinates.
    The function also computes the barycentric coordinates of the world point wrt the
    tetrahedrons from body A and body B.

    :param p:    The contact point in the model space of body B
    :param n:    The contact normal in the model space of body B
    :param XA:   The corner points of tetrahedron A in world coordinates.
    :param XB:   The corner points of tetrahedron B in world coordinates.
    :param X0B:  The corner points of tetrahedron B in material coordinates.
    :return:     The contact point/normal and barycentric coordinates wrt world space (n,p,omegaA, omegaB)

    """
    omegaB = BC.compute_barycentric_tetrahedron(X0B[0], X0B[1], X0B[2], X0B[3], p)
    p = np.dot(XB.T, omegaB)
    omegaA = BC.compute_barycentric_tetrahedron(XA[0], XA[1], XA[2], XA[3], p)
    D = XB[0:3] - XB[3]
    D0 = X0B[0:3] - X0B[3]
    n = D.dot(np.linalg.solve(D0, n))
    return p, n, omegaA, omegaB


def _compute_contacts(engine, stats, bodyA, bodyB, results, debug_on):
    """

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param bodyA:
    :param bodyB:
    :param results:     Triangle pairs. First column triangle idx from body A, second column is idx from body B.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary with profiling and timing measurements.
    """
    contact_optimization_timer = None
    model_space_update_timer = None
    contact_point_generation_timer = None
    if debug_on:
        model_space_update_timer = Timer("model_space_update")
        contact_optimization_timer = Timer("contact_optimization")
        contact_point_generation_timer = Timer("contact_point_generation")

    # Loop over all triangles from body A that are colliding with body B
    for k in range(len(results)):
        idx_triA, idx_triB = results[k]  # Get the triangle face indices
        idx_tetA = bodyA.owners[idx_triA][0]  # Get index of tetrahedron that f_a comes from
        idx_tetB = bodyB.owners[idx_triB][0]  # Get index of tetrahedron that f_b comes from

        if debug_on:
            contact_optimization_timer.start()
            model_space_update_timer.start()

        # Transform triangle A into model coordinates of body B.
        P = bodyA.x[bodyA.surface[idx_triA], :]  # Triangle face A vertices in world space
        XB = bodyB.x[bodyB.T[idx_tetB], :]  # Tetrahedron B vertices in world space
        X0B = bodyB.x0[bodyB.T[idx_tetB], :]  # Tetrahedron B vertices in material space
        P0 = _xform_triangle_to_model_space(
            P, XB, X0B
        )  # Vertices of triangle A in the body space of body B.

        if debug_on:
            model_space_update_timer.end()

        # We perform a Frank-Wolfe optimization algorithm, https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
        # This was shown to be very efficient in this paper: https://dl.acm.org/doi/10.1145/3384538
        #
        # First we initialize our optimization method for searching for deepest penetrating
        # point on the triangle from body A. This is done by first extracting the three
        # corner vertices of body A's triangle then we compute the norm of the gradients at
        # corner points of the triangle, and lastly we pick the corner point with the smallest
        # gradient norm.
        gradients = np.linalg.norm(
            [GRID.get_gradient(bodyB.grid, p_b) for p_b in P0], axis=1
        )
        x_i = P0[np.argmin(gradients)]

        # Next we are ready for performing the actual optimization, and we set up
        # a loop running for a maximum allowed number of iterations.
        for i in range(engine.params.contact_optimization_max_iterations):
            # TODO 2022-01-02 Kenny: The computation of the objectives repeats the same gradient computation three
            #  times. This could be optimized by simply storing the value of the norm of the gradient
            #  of x_i in a local variable.
            # Pick the triangle vertex 's_i' which minimizes the dot product with
            # the current gradient at 'x_i'. That is the vertex with "largest" descent.
            objectives = [np.dot(s_i, GRID.get_gradient(bodyB.grid, x_i)) for s_i in P0]
            vertex = np.argmin(objectives)
            s_i = P0[vertex]
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
        if debug_on:
            contact_optimization_timer.end()
            contact_point_generation_timer.start()
        # We have now optimized for the deepest penetrating point on the triangle, and now we can use this
        # point and the SDF from body B to generate a contact point between body A and body B. First we test
        # if our triangle point is even inside axis aligned bounding box (AABB) around the SDF of body B.
        if GRID.is_inside(bodyB.grid, x_i):
            # If we are inside the AABB then we look up the actual SDF value at the point. If the SDF value
            # is non-positive then we know we have a "true" contact point.
            phi = GRID.get_value(bodyB.grid, x_i)
            if phi <= engine.params.envelope:
                # In which case we can generate the contact point information. We use the SDF value as an
                # estimate of the penetration depth, and we take the gradient from the SDF at the contact
                # point as the normal value. However, we must remember that the whole optimization problem
                # was solved in the local body frame of body B. Hence, we must warp back the
                # contact point information into world space before we can report back the new contact point.
                gap = phi
                n = GRID.get_gradient(bodyB.grid, x_i)
                if V3.norm(n) > 0:
                    XA = bodyA.x[bodyA.T[idx_tetA], :]  # Tetrahedron A vertices in world space
                    p, n, omegaA, omegaB = _xform_contact_to_world(x_i, n, XA, XB, X0B)
                    cp = ContactPoint(
                        bodyB, bodyA,
                        idx_tetB, idx_tetA,
                        omegaB, omegaA,
                        p, V3.unit(n), gap
                    )
                    engine.contact_points.append(cp)
        if debug_on:
            contact_point_generation_timer.end()
    # Before we exit we just make sure we collect any stats and timings.
    if debug_on:
        if "model_space_update" not in stats:
            stats["model_space_update"] = 0
        stats["model_space_update"] += model_space_update_timer.total
        if "contact_optimization" not in stats:
            stats["contact_optimization"] = 0
        stats["contact_optimization"] += contact_optimization_timer.total
        if "contact_point_generation" not in stats:
            stats["contact_point_generation"] = 0
        stats["contact_point_generation"] += contact_point_generation_timer.total
    return stats


def _contact_determination(overlaps, engine, stats, debug_on):
    """

    :param overlaps:    A dictionary of triangles from one body that overlaps another body.
    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary with profiling and timing measurements.
    """
    contact_determination_timer = None
    if debug_on:
        contact_determination_timer = Timer("contact_determination", 8)
        contact_determination_timer.start()
    engine.contact_points = []
    for key, results in overlaps.items():
        # TODO 2022-12-31 Kenny: The code currently computes a lot of redundant contacts due
        #  to BVH traversal may return a triangle as part of several pairs. We only need
        #  the triangle pair information to warp a triangle from local space of one body
        #  into the local SDF space of the other body. However, we just need one pair where
        #  a specific triangle is part of, not all pairs where the triangle is part of.
        _compute_contacts(
            engine,
            stats,
            key[0],  # Body A
            key[1],  # Body B
            results,
            debug_on
        )
        _compute_contacts(
            engine,
            stats,
            key[1],  # Body B
            key[0],  # Body A
            results[
                :, [1, 0]
            ],  # Observe we swap columns because role of body A and body B is swapped.
            debug_on,
        )
    if debug_on:
        contact_determination_timer.end()
        stats["contact_determination"] = contact_determination_timer.elapsed
    return stats


def _contact_reduction(engine, stats, debug_on):
    """ During contact point computation it may happen that different colliding triangles of one body results
    in the same contact point locations wrt to the other signed distance field of the other body. Imagine a spiky
    polygonal cone pointing into a spherical shape. Here all triangles of the spike will result in the same
    contact point at the spiky tip. If the spiky polygonal cone has N triangle faces on the spike then we
     will have N redundant contact points. This redundancy is usually bad for other numerical sub-routines for
    computing impacts or contact forces. Hence, the purpose of this step is to eliminate redundant
    contact point information. One may think of this as a kind of post-filtering process to clean up the
     contact point information.

    Args:
        engine (Engine): The current engine instance we are working with.
        stats (dict):  A dictionary where to add more profiling and timing measurements.
        debug_on (bool): Boolean flag for toggling debug (aka profiling) info on and off.

    Returns:
        dict: A dictionary with profiling and timing measurements.
    """
    reduction_timer = None
    if debug_on:
        reduction_timer = Timer("contact_point_reduction", 8)
        reduction_timer.start()

    seen_ids = set()
    reduced_list = []

    for cp in engine.contact_points:
        uid = (cp.bodyA, cp.bodyB, tuple(cp.p))
        if uid not in seen_ids:
            reduced_list.append(cp)
            seen_ids.add(uid)

    engine.contact_points = reduced_list

    if debug_on:
        reduction_timer.end()
        stats["contact_point_reduction"] = reduction_timer.elapsed

    return stats


def run_collision_detection(engine, stats, debug_on):
    """
    This function invokes the whole collision detection pipeline on all the bodies
    currently active in the provided engine instance.

    It is assumed that state information of all bodies have been correctly updated to reflect the
    instant-of-time position where one wish to perform the collision detection at.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary with profiling and timing measurements.
    """
    # 2022-03-27 Kenny TODO: Interface should be reworked to take as input the global position and velocity
    #                    vectors, x, u of all bodies as input, as well as a look-ahead time-step. Further, the
    #                    method should return the contact point information as a return value, as well as a
    #                    next-safe time-step value. Using these changes will allow the collision detection module to
    #                    support continuous collision detection as well as discrete collision detection too.
    collision_detection_timer = None
    if debug_on:
        collision_detection_timer = Timer("collision_detection")
        collision_detection_timer.start()
    if engine.params.use_spatial_hashing:
        overlaps, stats = _spatial_hashing_narrow_phase(engine, stats, debug_on)
    else:
        stats = _update_bvh(engine, stats, debug_on)
        overlaps, stats = _narrow_phase(engine, stats, debug_on)
    stats = _contact_determination(overlaps, engine, stats, debug_on)
    stats = _contact_reduction(engine, stats, debug_on)
    if debug_on:
        collision_detection_timer.end()
        stats["collision_detection_time"] = collision_detection_timer.elapsed
    return stats
