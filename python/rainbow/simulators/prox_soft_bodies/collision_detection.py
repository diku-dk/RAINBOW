import rainbow.geometry.grid3 as GRID
import rainbow.geometry.kdop_bvh as BVH
import rainbow.geometry.barycentric as BC
from rainbow.simulators.prox_soft_bodies.types import *
import rainbow.cuda.collision_detection.compute_contacts as CUDA_COMPUTE_CONTACTS
from rainbow.util.timer import Timer
import numpy as np
from numba import cuda
from itertools import combinations


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


def _uniform_padding(listss, padding_value):
    """ Pad the listss to the same length, means that the length of each sub list is the same.

    Args:
        listss (List[List]): A nested list.
        padding_value (DataType): A padding value, which is used to pad the list.

    Returns:
        List[List]: the padded listss, which has the same length of each sub list.
    """
    valid_lists = [l for l in listss if l is not None]

    if len(valid_lists) == 0:
        return listss

    max_len = max(len(l) for l in valid_lists)

    return [np.pad(sub_list,
                   ((0, max_len - len(sub_list)), (0, 0)),
                   mode='constant',
                    constant_values=padding_value)
            for sub_list in valid_lists]


def _assemble_body_data_to_gpu(data_lists, bodyA, bodyB, triA, triB):
    """ Assemble body data to a data list for GPU computing.

    Args:
        data_lists (List): The data list for GPU computing.
        bodyA (SoftBody): A SofyBody instance.
        bodyB (SoftBody): A SofyBody instance.
        triA (int): The index of triangle in bodyA.
        triB (int): The index of triangle in bodyB.
    """
    data_lists['bodyA_idxs'].append(bodyA.idx)
    data_lists['bodyB_idxs'].append(bodyB.idx)
    data_lists['overlap_results'].append((triA, triB))
    data_lists['B_values'].append(bodyB.grid.values)
    data_lists['A_owners'].append(bodyA.owners)
    data_lists['B_owners'].append(bodyB.owners)
    data_lists['A_xs'].append(bodyA.x)
    data_lists['B_xs'].append(bodyB.x)
    data_lists['B_x0s'].append(bodyB.x0)
    data_lists['A_surfaces'].append(bodyA.surface)
    data_lists['A_Ts'].append(bodyA.T)
    data_lists['B_Ts'].append(bodyB.T)
    data_lists['B_grid_min_coords'].append(bodyB.grid.min_coord)
    data_lists['B_grid_max_coords'].append(bodyB.grid.max_coord)
    data_lists['B_grid_spacings'].append(bodyB.grid.spacing)
    data_lists['B_grid_Is'].append(bodyB.grid.I)
    data_lists['B_grid_Js'].append(bodyB.grid.J)
    data_lists['B_grid_Ks'].append(bodyB.grid.K)


def _contact_point_gpu(overlaps, engine, stats, debug_on):
    """ The GPU version of contact point computing, it flattens the data and send to GPU, then call the kernel function.

    Args:
        overlaps (dict): A dictionary of triangles from one body that overlaps another body.
        engine (Engine):  The current engine instance we are working with.
        stats (dict): A dictionary where to add more profiling and timing measurements.
        debug_on (bool): Boolean flag for toggling debug (aka profiling) info on and off.

    Returns:
        dict:  A dictionary with profiling and timing measurements.
    """
    contact_point_generation_timer = None
    if debug_on:
        contact_point_generation_timer = Timer("contact_point_generation")

    data_lists = {
        'bodyA_idxs': [],
        'bodyB_idxs': [],
        'overlap_results': [],
        'B_values': [],
        'A_owners': [],
        'B_owners': [],
        'A_xs': [],
        'B_xs': [],
        'B_x0s': [],
        'A_surfaces': [],
        'A_Ts': [],
        'B_Ts': [],
        'B_grid_min_coords': [],
        'B_grid_max_coords': [],
        'B_grid_spacings': [],
        'B_grid_Is': [],
        'B_grid_Js': [],
        'B_grid_Ks': []
    }

    for key, results in overlaps.items():
        for triA, triB in results:
            bodyA, bodyB = key
            _assemble_body_data_to_gpu(data_lists, bodyA, bodyB, triA, triB)
            _assemble_body_data_to_gpu(data_lists, bodyB, bodyA, triB, triA)

    data_length = len(data_lists["overlap_results"])
    if data_length == 0:
        return stats
    
    data_lists['A_owners'] = _uniform_padding(data_lists['A_owners'], -1)
    data_lists['B_owners'] = _uniform_padding(data_lists['B_owners'], -1)
    data_lists['A_xs'] = _uniform_padding(data_lists['A_xs'], -np.inf)
    data_lists['B_xs'] = _uniform_padding(data_lists['B_xs'], -np.inf)
    data_lists['B_x0s'] = _uniform_padding(data_lists['B_x0s'], -np.inf)
    data_lists['A_surfaces'] = _uniform_padding(data_lists['A_surfaces'], -1)
    data_lists['A_Ts'] = _uniform_padding(data_lists['A_Ts'], -1)
    data_lists['B_Ts'] = _uniform_padding(data_lists['B_Ts'], -1)

    type_map = {
        'bodyA_idxs': np.int32,
        'bodyB_idxs': np.int32,
        'overlap_results': np.int32,
        'B_values': np.float64,
        'A_owners': np.int32,
        'B_owners': np.int32,
        'A_xs': np.float64,
        'B_xs': np.float64,
        'B_x0s': np.float64,
        'A_surfaces': np.int32,
        'A_Ts': np.int32,
        'B_Ts': np.int32,
        'B_grid_min_coords': np.float64,
        'B_grid_max_coords': np.float64,
        'B_grid_spacings': np.float64,
        'B_grid_Is': np.int32,
        'B_grid_Js': np.int32,
        'B_grid_Ks': np.int32
    }

    # copy data to GPU
    d_data = {}
    for key, data in data_lists.items():
        array_data = np.array(data, dtype=type_map.get(key))
        d_data[f'd_{key}'] = cuda.to_device(array_data)

    # setting up GPU computing (grid and block)
    threads_per_block = engine.params.gpu_grid_size
    blocks_per_grid = (data_length + threads_per_block - 1) // threads_per_block

    # setting up result from GPU (data type and size)
    result_dtype = np.dtype([
        ('idx_tetB', np.int32),
        ('idx_tetA', np.int32),
        ('omegaB', (np.float64, 4)),
        ('omegaA', (np.float64, 4)),
        ('p', (np.float64, 3)),
        ('unit_n', (np.float64, 3)),
        ('gap', np.float64)
    ])
    result_gpu = cuda.device_array(data_length, dtype=result_dtype)

    # call GPU kernel function
    CUDA_COMPUTE_CONTACTS.contact_points_computing_kernel[blocks_per_grid, threads_per_block](
        d_data['d_bodyA_idxs'], d_data['d_bodyB_idxs'], d_data['d_overlap_results'],
        d_data['d_B_values'], d_data['d_A_owners'], d_data['d_B_owners'],
        d_data['d_A_xs'], d_data['d_B_xs'], d_data['d_B_x0s'], d_data['d_A_surfaces'],
        d_data['d_A_Ts'], d_data['d_B_Ts'],
        d_data['d_B_grid_min_coords'], d_data['d_B_grid_max_coords'], d_data['d_B_grid_spacings'],
        d_data['d_B_grid_Is'], d_data['d_B_grid_Js'], d_data['d_B_grid_Ks'],
        engine.params.contact_optimization_max_iterations, 
        engine.params.contact_optimization_tolerance,
        engine.params.envelope, 0.5, result_gpu)

    # wait for GPU data and copy to CPU
    cuda.synchronize() 
    result_to_cpu = result_gpu.copy_to_host() 

    ## generate contact points
    for res in result_to_cpu:
        if (res['idx_tetB'] < 0) and (res['idx_tetA'] < 0):
            continue
        cp = ContactPoint(
                bodyB, bodyA,
                res['idx_tetB'], res['idx_tetA'],
                res['omegaB'], res['omegaA'],
                res['p'], res['unit_n'], res['gap']
            )
        engine.contact_points.append(cp)

    if debug_on:
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

    ## contact points computing on GPU
    if cuda.is_available() and engine.params.use_gpu:
        _contact_point_gpu(overlaps, engine, stats, debug_on)

        if debug_on:
            contact_determination_timer.end()
            stats["contact_determination"] = contact_determination_timer.elapsed
        return stats
    
    # contact points computing on CPU, if GPU is not available or the flag is Flase
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
    """
    During contact point computation it may happen that different colliding triangles of one body results
    in the same contact point locations wrt to the other signed distance field of the other body. Imagine a spiky
    polygonal cone pointing into a spherical shape. Here all triangles of the spike will result in the same
    contact point at the spiky tip. If the spiky polygonal cone has N triangle faces on the spike then we
     will have N redundant contact points. This redundancy is usually bad for other numerical sub-routines for
    computing impacts or contact forces. Hence, the purpose of this step is to eliminate redundant
    contact point information. One may think of this as a kind of post-filtering process to clean up the
     contact point information.

     :param engine:      The current engine instance we are working with.
     :param stats:       A dictionary where to add more profiling and timing measurements.
     :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
     :return:            A dictionary with profiling and timing measurements.
    """
    reduction_timer = None
    if debug_on:
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
    stats = _update_bvh(engine, stats, debug_on)
    overlaps, stats = _narrow_phase(engine, stats, debug_on)
    stats = _contact_determination(overlaps, engine, stats, debug_on)
    stats = _contact_reduction(engine, stats, debug_on)
    if debug_on:
        collision_detection_timer.end()
        stats["collision_detection_time"] = collision_detection_timer.elapsed
    return stats
