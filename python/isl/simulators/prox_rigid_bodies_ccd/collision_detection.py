import isl.math.quaternion as Q
import isl.geometry.grid3 as GRID
import isl.geometry.kdop_bvh as BVH
from isl.simulators.prox_rigid_bodies_ccd.types import *
from itertools import combinations
from isl.util.timer import Timer
import numpy as np
import queue

def _update_bvh(dt, engine, stats, debug_on):
    """
    This function updates the swept bounding volume hierarchies of the rigid bodies to reflect the current
    world space geometry of the bodies. This is necessary to make sure that we later in the narrow phase
    collision detection function will be working on the geometry where it is actually correctly placed.
    in the world.

    :param dt:          The time-step to sweep the bounding volume.
    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary with profiling and timing measurements.
    """
    update_bvh_timer = None
    if debug_on:
        update_bvh_timer = Timer('update_bvh')
        update_bvh_timer.start()
    for body in engine.bodies.values():
        # Bounding volume hierarchy (BVH) traversal testing is done
        # in world space. Hence, before we refit the bounding volumes (BVs) of
        # the BVH trees we need to update the "geometry" that the BVH bounds to
        # reflect its current world space position. For rigid bodies this is simply
        # done by applying a rigid body transform to all the nodes of the triangle mesh.
        V_world = Q.rotate_array(body.q, body.shape.mesh.V) + body.r
        U_world = Q.rotate_array(Q.unit(body.q + (Q.prod(Q.from_vector3(body.w), body.q) * dt * 0.5)), body.shape.mesh.V) + body.r + body.v * dt
        # After we have updated the triangle mesh nodes we can invoke the refit
        # sub-routine from the BVH module.
        BVH.refit_bvh(V_world,
                      U_world,
                      body.shape.mesh.T,
                      body.bvh,
                      engine.params.K,
                      engine.params.envelope
                      )
    if debug_on:
        update_bvh_timer.end()
        stats['update_bvh'] = update_bvh_timer.elapsed
    return stats


def _narrow_phase(engine, stats, debug_on):
    """
    This function performs narrow phase collision detection between overlapping bodies in the engine. Narrow
    phase collision detection means looking more closely for potential overlapping triangle pairs between
    a pair of bodies.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A pair of dictionaries. The first is a dictionary where keys are overlapping bodies
                         and values are a list of potential colliding triangle pairs. A dictionary with
                         profiling and timing measurements.
    """
    narrow_phase_timer = None
    if debug_on:
        narrow_phase_timer = Timer('narrow_phase', 8)
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
        stats['narrow_phase'] = narrow_phase_timer.elapsed
        stats['number_of_overlaps'] = np.sum([len(results) for results in overlaps.values()])
    return overlaps, stats

def _compute_contacts_dcd(engine, stats, bodyA, bodyB, trianglesA, debug_on):
    """
    This function computes contacts between triangles from body A against the signed distance field of body B.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param bodyA:       Reference to body A.
    :param bodyB:       Reference to body B.
    :param trianglesA:  Array of triangles from body A that may be colliding with body B.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            Nothing.
    """
    model_space_update_timer = None
    contact_optimization_timer = None
    contact_point_generation_timer = None
    if debug_on:
        model_space_update_timer = Timer('model_space_update')
        contact_optimization_timer = Timer('contact_optimization')
        contact_point_generation_timer = Timer('contact_point_generation')
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
    if debug_on:
        model_space_update_timer.end()
    # Loop over all triangles from body A that are colliding with body B
    for t_a in trianglesA:
        if debug_on:
            contact_optimization_timer.start()
        # We perform a Frank-Wolfe optimization algorithm, https://en.wikipedia.org/wiki/Frank%E2%80%93Wolfe_algorithm
        # This was shown to be very efficient in this paper: https://dl.acm.org/doi/10.1145/3384538
        #
        # First we initialize our optimization method for searching for deepest penetrating
        # point on the triangle from body A. This is done by first extracting the three
        # corner vertices of body A's triangle
        p_bs = V_b[bodyA.shape.mesh.T[t_a], :]
        # then we compute the norm of the gradients at corner points of the triangle
        gradients = np.linalg.norm([GRID.get_gradient(bodyB.shape.grid, p_b) for p_b in p_bs], axis=1)
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
            objectives = [np.dot(s_i, GRID.get_gradient(bodyB.shape.grid, x_i)) for s_i in p_bs]
            vertex = np.argmin(objectives)
            s_i = p_bs[vertex]
            # Knowing that 's_i' has a "better" descent direction we update 'x_i' by "dragging" it
            # in the direction of 's_i'. The step-size that we update 'x_i' with is given by alpha.
            # We decrease the value of alpha as we iterate to ensure we do not overstep our minimizer.
            alpha = 2/(i+2)
            x_i = x_i + alpha*(s_i - x_i)
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
        if debug_on:
            contact_point_generation_timer.end()
    # Before we exit we just make sure we collect any stats and timings.
    if debug_on:
        if 'model_space_update' not in stats:
            stats['model_space_update'] = 0
        stats['model_space_update'] += model_space_update_timer.total
        if 'contact_optimization' not in stats:
            stats['contact_optimization'] = 0
        stats['contact_optimization'] += contact_optimization_timer.total
        if 'contact_point_generation' not in stats:
            stats['contact_point_generation'] = 0
        stats['contact_point_generation'] += contact_point_generation_timer.total


class Interval():
    def __init__(self, l, r):
        self.L = l
        self.R = r

    def __add__(self, other):
        a, b, c, d = self.L, self.R, other.L, other.R
        return Interval(a + c, b + d)

    def __sub__(self, other):
        a, b, c, d = self.L, self.R, other.L, other.R
        return Interval(a - d, b - c)

    def __mul__(self, other):
        if isinstance(other, float):
            return Interval(self.L * other, self.R * other)

        a, b, c, d = self.L, self.R, other.L, other.R
        if isinstance(other, Interval3):
            return Interval3(a * c, b * d)

        elif isinstance(other, self.__class__):
            return Interval(min(a*c, a*d, b*c, b*d),
                            max(a*c, a*d, b*c, b*d))

    def __div__(self, other):
        a, b, c, d = self.L, self.R, other.L, other.R
        # [c,d] cannot contain zero:
        if c*d <= 0:
            raise ValueError(f"Interval {other} cannot be denominator because it contains zero")
        return Interval(min(a/c, a/d, b/c, b/d),
                        max(a/c, a/d, b/c, b/d))

    def __contains__(self, key):
        return self.L <= key and key <= self.R

    def w(self):
        return self.R - self.L

    def __str__(self):
        return f"[{self.L}, {self.R}]"


class Interval3():
    def __init__(self, t, u, v):
        self.t = t
        self.u = u
        self.v = v

    def w(self):
        return max(self.t.w(), self.u.w(), self.v.w())

    def __str__(self):
        return f"[t: {self.t}, u: {self.u}, v: {self.v}]"


class VertexFace():
    def __init__(self,
                 v_t0, f_v0_t0, f_v1_t0, f_v2_t0,
                 v_t1, f_v0_t1, f_v1_t1, f_v2_t1):

        self.P_t0 = v_t0
        self.V_p = v_t1 - v_t0
        self.A_t0 = f_v0_t0
        self.V_a = f_v0_t1 - f_v0_t0
        self.B_t0 = f_v1_t0
        self.V_b = f_v1_t1 - f_v1_t0
        self.C_t0 = f_v2_t0
        self.V_c = f_v2_t1 - f_v2_t0

    def P(self, t):
        return self.P_t0 + t*self.V_p

    def v1(self, t):
        return self.A_t0 + t*self.V_a

    def v2(self, t):
        return self.B_t0 + t*self.V_b

    def v3(self, t):
        return self.C_t0 + t*self.V_c

    def N(self, t):
        return np.cross(self.v2(t) - self.v1(t), self.v3(t) - self.v1(t))

    def mapping(self, t, u, v):
        return self.P(t) - ((1 - u - v)*self.v1(t) + u*self.v2(t) + v*self.v3(t))

    def inclusion(self, I: Interval3):
        res = []
        for t in [I.t.L, I.t.R]:
            for u in [I.u.L, I.u.R]:
                for v in [I.v.L, I.v.R]:
                    res.append(self.mapping(t, u, v))
        x_max = max(res, key=lambda v: v[0])[0]
        y_max = max(res, key=lambda v: v[1])[1]
        z_max = max(res, key=lambda v: v[2])[2]
        x_min = min(res, key=lambda v: v[0])[0]
        y_min = min(res, key=lambda v: v[1])[1]
        z_min = min(res, key=lambda v: v[2])[2]
        return ((0 >= x_min and 0 <= x_max)
            and (0 >= y_min and 0 <= y_max)
            and (0 >= z_min and 0 <= z_max))


class EdgeEdge():
    def __init__(self,
                 p1_t0, p2_t0, p3_t0, p4_t0,
                 p1_t1, p2_t1, p3_t1, p4_t1):

        self.p1_t0 = p1_t0
        self.p1_v = p1_t1 - p1_t0
        self.p2_t0 = p2_t0
        self.p2_v = p2_t1 - p2_t0
        self.p3_t0 = p3_t0
        self.p3_v = p3_t1 - p3_t0
        self.p4_t0 = p4_t0
        self.p4_v = p4_t1 - p4_t0

    def P1(self, t):
        return self.p1_t0 + t*self.p1_v

    def p2(self, t):
        return self.p2_t0 + t*self.p2_v

    def p3(self, t):
        return self.p2_t0 + t*self.p3_v

    def p4(self, t):
        return self.p4_t0 + t*self.p4_v

    def mapping(self, t, u, v):
        return ((1 - u)*self.p1(t) + u*self.p2(t)) - ((1 - v)*self.p3(t) + v*self.p4(t)) 

    def inclusion(self, I: Interval3):
        res = []
        for t in [I.t.L, I.t.R]:
            for u in [I.u.L, I.u.R]:
                for v in [I.v.L, I.v.R]:
                    res.append(self.mapping(t, u, v))
        x_max = max(res, key=lambda v: v[0])[0]
        y_max = max(res, key=lambda v: v[1])[1]
        z_max = max(res, key=lambda v: v[2])[2]
        x_min = min(res, key=lambda v: v[0])[0]
        y_min = min(res, key=lambda v: v[1])[1]
        z_min = min(res, key=lambda v: v[2])[2]
        return ((0 >= x_min and 0 <= x_max)
            and (0 >= y_min and 0 <= y_max)
            and (0 >= z_min and 0 <= z_max))


def split_interval(I: Interval3):
    t_m = (I.t.L + I.t.R) / 2
    u_m = (I.u.L + I.u.R) / 2
    v_m = (I.v.L + I.v.R) / 2
    t_1, t_2 = Interval(I.t.L, t_m), Interval(t_m, I.t.R)
    u_1, u_2 = Interval(I.u.L, u_m), Interval(u_m, I.u.R)
    v_1, v_2 = Interval(I.v.L, v_m), Interval(v_m, I.v.R)

    return [
                Interval3(t_1, u_1, v_1),
                Interval3(t_1, u_2, v_1),
                Interval3(t_1, u_1, v_2),
                Interval3(t_1, u_2, v_2),
                Interval3(t_2, u_1, v_1),
                Interval3(t_2, u_2, v_1),
                Interval3(t_2, u_1, v_2),
                Interval3(t_2, u_2, v_2)
            ]


def solve_interval(I_0: Interval3, g, sigma):
    res = []
    l = 0

    q = queue.Queue()
    q.put(I_0)

    while q.qsize() != 0:
        I = q.get()
        I_g = g.inclusion(I)
        if I_g:
            # print(I.w())
            if I.w() < sigma:
                res.append(I)
            else:
                Is = split_interval(I)
                for i in Is:
                    q.put(i)
        l = l + 1

    min_res = min(res, key=lambda tuv: tuv.t.L) if len(res) > 0 else None
    return min_res
    # return min(res, key=lambda tuv: tuv.t.L) if len(res) > 0 else None


def _compute_vertex_face_ccd(v_t0, f_v0_t0, f_v1_t0, f_v2_t0,
                             v_t1, f_v0_t1, f_v1_t1, f_v2_t1):

    vf = VertexFace(v_t0, f_v0_t0, f_v1_t0, f_v2_t0,
                    v_t1, f_v0_t1, f_v1_t1, f_v2_t1)

    res = solve_interval(Interval3(Interval(0, 1), Interval(0, 1), Interval(0, 1)), vf, 0.000001)
    if res is not None:
        return [res.t, vf.P(res.t.L), vf.N(res.t.L), 0]
    return None

def _compute_edge_edge_ccd(p1_t0, p2_t0, p3_t0, p4_t0,
                           p1_t1, p2_t1, p3_t1, p4_t1):

    ee = EdgeEdge(p1_t0, p2_t0, p3_t0, p4_t0,
                  p1_t1, p2_t1, p3_t1, p4_t1)

    return solve_interval(Interval3(Interval(0, 1), Interval(0, 1), Interval(0, 1)), ee, 0.000001)


def _compute_contacts(engine, stats, dt, bodyA, bodyB, triangles, debug_on):
    """
    This function computes time of impacts and contacts between triangles from body A and body B using continuous collision detecton.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param dt:          Time-step
    :param bodyA:       Reference to body A.
    :param bodyB:       Reference to body B.
    :param trianglesA:  Array of triangles from body A that may be colliding with body B.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            Nothing.
    """
    # Make BVH traversel return start and end points instead
    toi = Interval(np.Infinity, np.Infinity)
    mesh_A_0 = Q.rotate_array(bodyA.q, bodyA.shape.mesh.V) + bodyA.r
    mesh_B_0 = Q.rotate_array(bodyB.q, bodyB.shape.mesh.V) + bodyB.r
    mesh_A_1 = Q.rotate_array(Q.unit(bodyA.q + (Q.prod(Q.from_vector3(bodyA.w), bodyA.q) * dt * 0.5)), bodyA.shape.mesh.V) + bodyA.r + bodyA.v * dt
    mesh_B_1 = Q.rotate_array(Q.unit(bodyB.q + (Q.prod(Q.from_vector3(bodyB.w), bodyB.q) * dt * 0.5)), bodyB.shape.mesh.V) + bodyB.r + bodyB.v * dt
    # res = []
    for triangleA, triangleB in triangles:
        f_a_t0 = mesh_A_0[bodyA.shape.mesh.T[triangleA]]
        f_a_t1 = mesh_A_1[bodyA.shape.mesh.T[triangleA]]
        f_b_t0 = mesh_B_0[bodyB.shape.mesh.T[triangleB]]
        f_b_t1 = mesh_B_1[bodyB.shape.mesh.T[triangleB]]
        # print("A", f_a_t0, f_a_t1)
        # print("B", f_b_t0, f_b_t1)
        for i in range(3):
            v_a_t0 = f_a_t0[i]
            v_a_t1 = f_a_t1[i]
            v_b_t0 = f_b_t0[i]
            v_b_t1 = f_b_t1[i]
            a_vf = _compute_vertex_face_ccd(v_a_t0, f_b_t0[0], f_b_t0[1], f_b_t0[2], v_a_t1, f_b_t1[0], f_b_t1[1], f_b_t1[2])
            b_vf = _compute_vertex_face_ccd(v_b_t0, f_a_t0[0], f_a_t0[1], f_a_t0[2], v_b_t1, f_a_t1[0], f_a_t1[1], f_a_t1[2])
            if a_vf is not None:
                dti = a_vf[0]*dt
                toi = min(toi, dti, key=lambda t: t.L)
                cp = ContactPoint(bodyA, bodyB, a_vf[1], V3.unit(a_vf[2]), a_vf[3], dti.L)
                engine.contact_points.append(cp)
                # res.append(a_vf)
            if b_vf is not None:
                dti = b_vf[0]*dt
                toi = min(toi, dti, key=lambda t: t.L)
                cp = ContactPoint(bodyA, bodyB, b_vf[1], V3.unit(b_vf[2]), b_vf[3], dti.L)
                engine.contact_points.append(cp)
                # res.append(b_vf)
            # print(f"{toi_a=}, {toi_b=}")
            # toi = min(toi, toi_a_vf)
            # toi = min(toi, toi_b_vf)
            # _compute_edge_edge_ccd()
        # for i in range(3):
        #     for j in range(3):
        #         p1_t0, p2_t0 = f_a_t0[i], f_a_t0[(i+1) % 3]
        #         p3_t0, p4_t0 = f_b_t0[j], f_b_t0[(j+1) % 3]
        #         p1_t1, p2_t1 = f_a_t1[i], f_a_t1[(i+1) % 3]
        #         p3_t1, p4_t1 = f_b_t1[j], f_b_t1[(j+1) % 3]
        #         ee = _compute_edge_edge_ccd(p1_t0, p2_t0, p3_t0, p4_t0,
        #                                     p1_t1, p2_t1, p3_t1, p4_t1)
                # if ee is not None:
                #     # res.append(ee)

    # print("final", toi)
    return toi

def _contact_determination(dt, overlaps, engine, stats, debug_on):
    """
    This function performs determination of contact points between all pairs of overlapping bodies. The function
     essentially post-process the overlap-data that was computed by the narrow-phase collision detection function.

    :param dt:          Time-step
    :param overlaps:    A dictionary where keys are overlapping bodies and values are a list of potential
                        colliding triangle pairs.
    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary with profiling and timing measurements.
    """
    contact_determination_timer = None
    if debug_on:
        contact_determination_timer = Timer('contact_determination', 8)
        contact_determination_timer.start()
    engine.contact_points = []
    toi = Interval(dt, dt)
    for key, results in overlaps.items():
        dti = _compute_contacts(engine,
                            stats,
                            dt,
                            key[0],  # Body A
                            key[1],  # Body B
                            results, # All triangles from A and B that may collide
                            debug_on
                            )
        toi = min(toi, dti, key=lambda t: t.L)

    if debug_on:
        contact_determination_timer.end()
        stats['contact_determination'] = contact_determination_timer.elapsed
    return stats, toi


def _contact_reduction(toi, engine, stats, debug_on):
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
        reduction_timer = Timer('contact_point_reduction', 8)
        reduction_timer.start()

    reduced_list = []

    for cp in engine.contact_points:
        if cp.toi <= toi.R:
            reduced_list.append(cp)

    engine.contact_points = reduced_list

    # TODO 2020-09-07 Kristian: This brute force implementation can be implemented better
    reduced_list = []
    for cp1 in engine.contact_points:
        unique = True
        for cp2 in reduced_list:
            if {cp1.bodyA, cp1.bodyB} == {cp2.bodyA, cp2.bodyB} and (cp1.p == cp2.p).all():
                unique = False
        if unique:
            reduced_list.append(cp1)
    engine.contact_points = reduced_list
    if debug_on:
        reduction_timer.end()
        stats['contact_point_reduction'] = reduction_timer.elapsed
    return stats


def run_collision_detection(dt, engine, stats, debug_on):
    """
    This function invokes the whole continuous collision detection pipeline on all the bodies
    currently active in the provided engine instance.

    It is assumed that state information of all bodies have been correctly updated to reflect the
    instant-of-time velocities of the rigid bodies one wish to perform the continuous collision detection for.

    :param dt:          The time-step to perform continuous collision detection for.
    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary with profiling and timing measurements.
    """
    collision_detection_timer = None
    if debug_on:
        collision_detection_timer = Timer('collision_detection')
        collision_detection_timer.start()
    stats = _update_bvh(dt, engine, stats, debug_on)
    overlaps, stats = _narrow_phase(engine, stats, debug_on)
    stats, toi = _contact_determination(dt, overlaps, engine, stats, debug_on)
    stats = _contact_reduction(toi, engine, stats, debug_on)
    if debug_on:
        collision_detection_timer.end()
        stats['collision_detection_time'] = collision_detection_timer.elapsed
    return stats, toi.L
