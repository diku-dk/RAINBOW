import rainbow.simulators.prox_soft_bodies.mechanics as MECH
import rainbow.simulators.prox_soft_bodies.collision_detection as CD
import rainbow.simulators.proximal_contact.prox_solvers as CONTACT_SOLVERS
import rainbow.math.vector3 as V3
import rainbow.math.matrix3 as M3
import numpy as np
import scipy.sparse as sparse
from typing import List
from rainbow.simulators.prox_soft_bodies.types import Engine
from rainbow.util.timer import Timer


class ElementArrayUtil:
    """
    This class contains sub-routines that is helpful for manipulating or using matrix
    element arrays.

    When applying FEM one usually gets a small local element matrix for each element.
    If the elements has N nodes, and we have 3 DOF then the FEM process will create a
    matrix of 3 N x 3 N dimension. However, we store this conceptually as N-by-N matrix
    of 3-by-3 matrices.

    For an example of how to compute an element matrix array
    see '~rainbow.simulators.prox_soft_bodies.solver.compute_element_mass_matrix'

    The reason for this is that it is more convenient to index the 3-by-3 matrices
    corresponding to a given pair of node indices (i,j).
    """

    @staticmethod
    def prod_mat_vec(T, A_array, x):
        """
        Computes the matrix vector product of a matrix given as an element array representation
        and a vector. This function allows to compute the matrix vector product without
        performing an assembly of the A matrix. The idea is that one can easily compute
        the value of A*x without having to do the assembly of A. If the mesh has K tetrahedrons
        and N nodes then x is 3N (or N-by-3) and A is a Kx4x4x3x3

        :param T:       The tetrahedral elements of the mesh.
        :param A_array: The element array representation of the A-matrix.
        :param x:       The vector to multiply with A.
        :return:        The value of the product A*x
        """
        y = np.zeros_like(x)
        if len(x.shape) == 1:
            for e in range(len(T)):
                for i in range(4):
                    for j in range(4):
                        i_offset = T[e, i] * 3
                        j_offset = T[e, j] * 3
                        y[i_offset : i_offset + 3] += np.dot(
                            A_array[e, i, j], x[j_offset : j_offset + 3]
                        )
        else:
            for e in range(len(T)):
                for i in range(4):
                    for j in range(4):
                        y[T[e, i]] += np.dot(A_array[e, i, j], x[T[e, j]])
        return y

    @staticmethod
    def assembly_coo(T, A_array):
        """
        Assemble element matrices (local elements) into a global matrix.

        :param T:       The tetrahedral elements.
        :param A_array: The element matrix array that should be assembled into a matrix.
        :return:        Three arrays, first is row index, second column index, and third is value.
        """
        lut = {}
        for e in range(len(T)):
            for i in range(4):
                for j in range(4):
                    # Local node indices converted to global node indices
                    key = (T[e, i], T[e, j])
                    # Check if we have seen this nodal pair before and then accumulate values if so
                    if key in lut:
                        lut[key] += A_array[e, i, j]
                    else:
                        lut[key] = np.copy(A_array[e, i, j])
        # Now we have accumulated all the element-wise matrices in the
        row = np.full(len(lut) * 9, -1, dtype=np.int32)
        col = np.full(len(lut) * 9, -1, dtype=np.int32)
        data = np.zeros(len(lut) * 9, dtype=np.float64)
        s = 0
        for key, value in lut.items():
            for i in range(3):
                for j in range(3):
                    row[s] = key[0] * 3 + i
                    col[s] = key[1] * 3 + j
                    data[s] = value[i, j]
                    s += 1
        return row, col, data

    @staticmethod
    def assembly_csr(N, T, A_array):
        """
        Assemble element matrices (local elements) into a global matrix.

        :param N:       Total number of nodes in the tetrahedral mesh (should be equal to max(T)+1)
        :param T:       The tetrahedral elements.
        :param A_array: The element matrix array that should be assembled into a matrix.
        :return:        A CSR sparse matrix holding the assembled element matrices.
        """
        row, col, data = ElementArrayUtil.assembly_coo(T, A_array)
        A = sparse.csr_matrix((data, (row, col)), shape=(N * 3, N * 3))
        return A


class Native:
    """
    This class contains functions for computing forces and mass/damping matrices.
    The functions are "hardwired" to a linear tetrahedral mesh data structure. As
    such the functions uses the raw native face-index array interface for passing
    arguments and values.

    As such the functions have no notion of a "soft-body" instead they just see
    a bunch of tetrahedral elements.
    """

    @staticmethod
    def compute_outward_face_vectors(x, T):
        """
        Computes outward pointing face vectors for each tetrahedron.

        Observe that outward face vectors are the negative gradient of the shape functions.

        :param x:    The current vertex coordinates to use for the face-vector computation.
        :param T:    The tetrahedron array.
        :return:     A K-by-3-by-4 array. One 3-by-4 array for each tetrahedron. Each
                     column corresponds to one face vector.
        """
        gradN = np.zeros((len(T), 3, 4), dtype=np.float64)
        for e in range(len(T)):
            i, j, k, m = T[e]
            ri = x[i]
            rj = x[j]
            rk = x[k]
            rm = x[m]
            # Compute edge vectors
            u_ji = rj - ri
            u_kj = rk - rj
            u_ik = ri - rk
            u_mi = rm - ri
            u_mj = rm - rj
            # The division by 6 is to avoid computations later on.
            # The N-vectors are used to evaluate an integral like this int_A (1/3) P N dA
            # Now remember that N given as edge cross product u and v will have 2 time the area A.
            # Therefore int_A ... dA =  1/3 A P N =  1/6 P cross(u,v). Hence, we just define N as
            # cross(u,v)/6 and in our implementations we have int_A ... dA = P N
            gradN[e, :, 0] = np.cross(u_kj, u_mj) / 6  # Ni, face normal of j,k, and m
            gradN[e, :, 1] = np.cross(u_ik, u_mi) / 6  # Nj, face normal of i,k, and m
            gradN[e, :, 2] = np.cross(u_ji, u_mi) / 6  # Nk, face normal of i,j, and m
            gradN[e, :, 3] = np.cross(u_kj, u_ji) / 6  # Nm, face normal of i,j, and k
        return gradN

    @staticmethod
    def compute_D(x, T):
        """
        Computes the spatial coordinate edge-vector matrix for each tetrahedron.

        :param x:       The deformed spatial coordinates of the vertices in the object.
        :param T:       Tetrahedral elements of the object.
        :return:        A K array of 3-by-3 arrays, each entry holds the inverted material
                        coordinates edge-vector matrix.
        """
        D = np.zeros((len(T), 3, 3), dtype=np.float64)
        for e in range(len(T)):
            # Extract material coordinates of tetrahedron
            p = x[T[e]]
            # Compute edge vectors of the tetrahedron
            u_ji = p[1] - p[0]
            u_ki = p[2] - p[0]
            u_mi = p[3] - p[0]
            # Verify that the tetrahedron is well-defined
            if np.dot(u_mi, np.cross(u_ji, u_ki)) <= 0:
                raise RuntimeError("compute_D(): Degenerate tetrahedron found", e)
            # Create edge-vector matrix
            D[e] = M3.make(
                u_ji[0],
                u_ki[0],
                u_mi[0],
                u_ji[1],
                u_ki[1],
                u_mi[1],
                u_ji[2],
                u_ki[2],
                u_mi[2],
            )
        return D

    @staticmethod
    def compute_inverse_D0(x0, T):
        """
        Computes the inverted material coordinate edge-vector matrix for each tetrahedron.

        :param x0:       The un-deformed material coordinates of the vertices in the object.
        :param T:        Tetrahedral elements of the object.
        :return:         A K array of 3-by-3 arrays, each entry holds the inverted material
                         coordinates edge-vector matrix.
        """
        D0 = Native.compute_D(x0, T)
        invD0 = np.zeros((len(T), 3, 3), dtype=np.float64)
        for e in range(len(T)):
            invD0[e] = np.linalg.inv(D0[e])
        return invD0

    @staticmethod
    def compute_deformation_gradient(x, T, invD0):
        """
        Computes the deformation gradient.

        :param x:          The current spatial nodal positions.
        :param T:          The tetrahedral elements.
        :param invD0:      The inverse material edge-matrix.
        :return:           The deformation gradients.
        """
        D = Native.compute_D(x, T)
        F = np.zeros((len(T), 3, 3), dtype=np.float64)
        for e in range(len(T)):
            F[e] = np.dot(D[e], invD0[e])
        return F

    @staticmethod
    def compute_mass_element_array(rho, vol, T, is_lumped):
        """
        The mass matrix for a single tetrahedral element consist of 4-by-4 array of 3-by-3 matrices. This
        function computes these element matrices and store then in one long array.

        :param rho:         The mass density of the tetrahedral elements.
        :param vol:         The volume of the tetrahedral elements.
        :param T:           The tetrahedral elements.
        :param is_lumped:   Flag to turn lumping on.
        :return:            If there i K elements then this is a K-by-( 4-by-4 by 3-by-3) array of
                            nodal mass matrices.
        """
        NN = np.zeros((4, 4, 3, 3), dtype=np.float64)
        if is_lumped:
            # Lumped mass matrix
            NN[0, 0] = np.identity(3) / 4
            NN[1, 1] = np.identity(3) / 4
            NN[2, 2] = np.identity(3) / 4
            NN[3, 3] = np.identity(3) / 4
        else:
            # Consistent mass matrix
            # Me = rho V / 20  *  [ 2*I_3x3    I_3x3   I_3x3   I_3x3;
            #                         I_3x3  2*I_3x3   I_3x3   I_3x3;
            #                         I_3x3    I_3x3 2*I_3x3   I_3x3;
            #                         I_3x3    I_3x3   I_3x3 2*I_3x3;]
            for i in range(4):
                for j in range(i + 1, 4):
                    NN[i, j] = np.identity(3) / 20
                    NN[j, i] = np.identity(3) / 20
            NN[0, 0] = np.identity(3) / 10
            NN[1, 1] = np.identity(3) / 10
            NN[2, 2] = np.identity(3) / 10
            NN[3, 3] = np.identity(3) / 10
        M_array = np.zeros((len(T), 4, 4, 3, 3), dtype=np.float64)
        for e in range(len(T)):
            M_array[e] = NN * rho * vol[e]
        return M_array

    @staticmethod
    def compute_damping_element_array(c, vol, T, is_lumped):
        """
        The damping matrix for a single tetrahedral element consist of 4-by-4 array of 3-by-3 matrices. This
        function computes these element matrices and store then in one long array.

        :param c:           The linear damping coefficient.
        :param vol:         The volume of the tetrahedral elements.
        :param T:           The tetrahedral elements.
        :param is_lumped:   Flag to turn lumping on.
        :return:            If there i K elements then this is a K-by-( 4-by-4 by 3-by-3) array
                            of nodal mass matrices.
        """
        NN = np.zeros((4, 4, 3, 3), dtype=np.float64)
        if is_lumped:
            # Lumped damping matrix
            NN[0, 0] = np.identity(3) / 4
            NN[1, 1] = np.identity(3) / 4
            NN[2, 2] = np.identity(3) / 4
            NN[3, 3] = np.identity(3) / 4
        else:
            # Consistent damping matrix
            # Me = rho V / 20  *  [ 2*I_3x3    I_3x3   I_3x3   I_3x3;
            #                         I_3x3  2*I_3x3   I_3x3   I_3x3;
            #                         I_3x3    I_3x3 2*I_3x3   I_3x3;
            #                         I_3x3    I_3x3   I_3x3 2*I_3x3;]
            for i in range(4):
                for j in range(i + 1, 4):
                    NN[i, j] = np.identity(3) / 20
                    NN[j, i] = np.identity(3) / 20
            NN[0, 0] = np.identity(3) / 10
            NN[1, 1] = np.identity(3) / 10
            NN[2, 2] = np.identity(3) / 10
            NN[3, 3] = np.identity(3) / 10
        C_array = np.zeros((len(T), 4, 4, 3, 3), dtype=np.float64)
        for e in range(len(T)):
            C_array[e] = NN * c * vol[e]
        return C_array

    @staticmethod
    def compute_traction_forces(x, traction_conditions):
        """
        This function computes the traction forces that act on an object. The load on the surface
        triangles are considered to be constant in world space. Hence, one can think of this as
        always pulling/pushing in the same direction while the object is deforming.

        :param x:                       The spatial coordinates of the object where the traction
                                        forces are to be computed for.
        :param traction_conditions:     A list of traction conditions
        :return:                        A vector for the whole object where each element holds the nodal
                                        force to apply to the corresponding node.
        """
        Ft = np.zeros_like(x, dtype=np.float64)
        Id = M3.identity()  # Just for convenience to make code more readable.
        for traction in traction_conditions:
            x_i = x[traction.i]
            x_j = x[traction.j]
            x_k = x[traction.k]
            A_vec = np.cross((x_j - x_i), (x_k - x_i)) / 2
            A = np.linalg.norm(A_vec)
            # --- Solving integral of products of shape functions for iso-parametric
            # --- linear triangle element
            #
            #    syms x y real
            #     Ni = 1 - x - y
            #     Nj = y
            #     Nk = x
            #     N = [Ni Nj Nk]
            #     A = N'* N
            #     AA = int(A,x,0,1-y)
            #     L = int(AA,y,0,1)
            #
            #  We find the load (nodal traction distribution)  matrix
            #
            #   L = 1 / 24  *  [ 2*I_3x3    I_3x3   I_3x3;
            #                      I_3x3  2*I_3x3   I_3x3;
            #                      I_3x3    I_3x3 2*I_3x3; ]
            # --- The spatial load force
            #
            #     lf(f) = A(f) * L * [li; lj; lk]
            #
            #   where load = (li,lj,lk) is the applied surface traction.
            load = traction.traction
            Ft[traction.i] += (A / 24) * (
                2 * np.dot(Id, load) + np.dot(Id, load) + np.dot(Id, load)
            )
            Ft[traction.j] += (A / 24) * (
                np.dot(Id, load) + 2 * np.dot(Id, load) + np.dot(Id, load)
            )
            Ft[traction.k] += (A / 24) * (
                np.dot(Id, load) + np.dot(Id, load) + 2 * np.dot(Id, load)
            )
        return Ft

    @staticmethod
    def compute_elastic_forces(x, T, gradN0, F, lambda_, mu_, pk1_stress):
        """
        Compute elastic forces.

        Recall that elastic forces are Fe = - P grad phi, where grad phi is the gradient of the shape function.
        For linear elements we have that (grad phi) is opposite the outward face normal of the opposing
        face s and has magnitude equal to area that face.

        :param x:              The current spatial position.
        :param T:              The tetrahedral elements.
        :param gradN0:         Outward face vectors (precomputed).
        :param F:              The deformation gradients.
        :param lambda_:        Lamé first parameter.
        :param mu_:            Lamé second parameter.
        :param pk1_stress:     The constitutive model to use.
        :return:               The elastic forces.
        """
        Fe = np.zeros((len(x), 3), dtype=np.float64)
        for e in range(len(T)):
            i, j, k, m = T[e]
            P = pk1_stress(F[e], lambda_, mu_)
            Fe[i] += P.dot(gradN0[e, :, 0])
            Fe[j] += P.dot(gradN0[e, :, 1])
            Fe[k] += P.dot(gradN0[e, :, 2])
            Fe[m] += P.dot(gradN0[e, :, 3])
        return Fe

    @staticmethod
    def compute_elastic_energy(vol0, F, lambda_, mu_, energy_density):
        """
        Compute the elastic energy.

        :param vol0:
        :param F:
        :param lambda_:
        :param mu_:
        :param energy_density:
        :return:
        """
        energy = 0
        # Below we use a FVM approach to compute energies per element. That is we use mid-point rule approximation.
        for e in range(len(vol0)):
            psi_mid = energy_density(F[e], lambda_, mu_)
            energy += psi_mid * vol0[e]
        return energy

    @staticmethod
    def compute_damping_forces(T, Carray, u):
        """
        Compute the damping forces.

        :param T:        The tetrahedral elements.
        :param Carray:   The damping element matrix array.
        :param u:        The nodal velocities, dimension is N-by-3 where N is number of nodes.
        :return:         The damping force vector, will be dimension N-by-3 where N is number of nodes.
        """
        Fd = ElementArrayUtil.prod_mat_vec(T, Carray, u)
        return -Fd

    @staticmethod
    def compute_gravity_forces(N, T, vol0, rho, g):
        """
        Compute gravity forces.

        :param N:        The total number of nodes in the mesh.
        :param T:        The tetrahedral elements.
        :param vol0:     The material coordinate volumes of the elements.
        :param rho:      The mass density value.
        :param g:        The gravitational acceleration vector. If z-axis is up then g = (0,0,-9.81)
        :return:         The gravitational force vector, will be dimension N-by-3.
        """
        Fg = np.zeros((N, 3), dtype=np.float64)
        for e in range(len(T)):
            i, j, k, m = T[e]
            dF = (rho * vol0[e] / 4.0) * g
            Fg[i] += dF
            Fg[j] += dF
            Fg[k] += dF
            Fg[m] += dF
        return Fg

    @staticmethod
    def apply_dirichlet_conditions(dirichlet_conditions, A, b) -> None:
        """
         This function applies the given dirichlet conditions to a linear system A x = b.

        This function is specifically targeting lumped systems where the coefficient matrix A is basically
        a 3-by-3 blocked diagonal matrix. That we can store as an array of 3-by-3 matrices.

        :param dirichlet_conditions:  A list of dirichlet conditions
        :param A:                     The coefficient matrix of the linear system. An N-by-3-by-3 array.
        :param b:                     The right-hand side vector of the linear system.
        """
        for bc in dirichlet_conditions:
            idx = bc.idx * 3
            b[idx : idx + 3] = bc.value
            # TODO 2020-09-18 Kenny: CSR may not be efficient
            #  for setting values like this. We might have
            #  to use a different sparse matrix format for this?
            A[idx : idx + 3, :] = 0
            A[idx : idx + 3, idx: idx + 3] = np.identity(3)


def get_friction_coefficient_vector(engine):
    """
    This function extracts proper friction coefficient values from the materials'
    interaction library and set them up in an array with an entry for each contact.

    :param engine:  A reference to the engine holding coefficient of friction values.
    :return:        A K dimensional vector holding the coefficient of friction value
                    for each contact where K is the number of contacts.
    """
    K = len(engine.contact_points)
    mu = np.zeros(K, dtype=np.float64)
    for k in range(K):
        cp = engine.contact_points[k]
        interaction = engine.surfaces_interactions.get_interaction(
            cp.bodyA.material_description.name, cp.bodyB.material_description.name
        )
        mu[k] = interaction.mu[0]
    return mu


def get_largest_penetration_error(engine):
    """
    This function examines the collision detection information stored in the engine
    and measures the largest penetration error.

    :param engine:    The current engine instance we are working with.
    :return:          The largest penetration error. It will be non-negative. The
                      larger the value the worse the error.
    """
    penetration = 0
    for cp in engine.contact_points:
        penetration = min(penetration, cp.g)
    return -penetration


def compute_jacobian_matrix(engine, stats, debug_on):
    """
    Compute the global contact Jacobian matrix.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            The contact Jacobian matrix.
    """
    timer = None
    if debug_on:
        timer = Timer("compute_jacobian_matrix")
        timer.start()
    K = len(engine.contact_points)
    N = engine.number_of_nodes

    # For each contact we have 8 non-zero blocks, each block has 4 rows and 3 columns,
    # That gives us 96 floats per contact.
    data = np.zeros(K * 96, dtype=np.float64)
    row = np.zeros(K * 96, dtype=np.int32)
    col = np.zeros(K * 96, dtype=np.int32)

    next_entry = 0
    for k in range(K):
        cp = engine.contact_points[k]
        # Obtain global node indices of the two tetrahedrons A and B
        iA, jA, kA, mA = cp.bodyA.T[cp.idx_tetA] + cp.bodyA.offset
        iB, jB, kB, mB = cp.bodyB.T[cp.idx_tetB] + cp.bodyB.offset
        # Compute the contact frame, the fourth row is intended for friction
        # torque, currently no need for this in soft bodies, so it will just be
        # zero here.
        vs, vt, vn = V3.make_orthonormal_vectors(cp.n)
        C = np.array([vn, vs, vt, V3.zero()], dtype=np.float64)
        # Compute the Jacobian blocks for the k-th contact
        JiA = -C.dot(np.eye(3) * cp.omegaA[0])
        JjA = -C.dot(np.eye(3) * cp.omegaA[1])
        JkA = -C.dot(np.eye(3) * cp.omegaA[2])
        JmA = -C.dot(np.eye(3) * cp.omegaA[3])
        JiB = C.dot(np.eye(3) * cp.omegaB[0])
        JjB = C.dot(np.eye(3) * cp.omegaB[1])
        JkB = C.dot(np.eye(3) * cp.omegaB[2])
        JmB = C.dot(np.eye(3) * cp.omegaB[3])
        # Fill in values into the global Jacobian matrix.
        for i in range(4):
            row_idx = k * 4 + i
            col_idx = iA * 3
            for j in range(3):
                data[next_entry] = JiA[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = jA * 3
            for j in range(3):
                data[next_entry] = JjA[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = kA * 3
            for j in range(3):
                data[next_entry] = JkA[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = mA * 3
            for j in range(3):
                data[next_entry] = JmA[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = iB * 3
            for j in range(3):
                data[next_entry] = JiB[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = jB * 3
            for j in range(3):
                data[next_entry] = JjB[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = kB * 3
            for j in range(3):
                data[next_entry] = JkB[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = mB * 3
            for j in range(3):
                data[next_entry] = JmB[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
    J = sparse.csr_matrix((data, (row, col)), shape=(K * 4, N * 3))
    if debug_on:
        timer.end()
        stats["compute_jacobian_matrix"] = timer.elapsed
    return J


def set_velocity_vector(u, engine) -> None:
    """
    This function sets the body velocity vector.

    :param u:         The concatenated velocity vector for the whole simulation.
    :param engine:    The current engine instance we are working with.
    :return:          Nothing.
    """
    for body in engine.bodies.values():
        body.u = u[body.offset : body.offset + len(body.u)]


def set_position_vector(x, engine) -> None:
    """
    This function sets the body position vectors.

    :param x:         The position that should be set.
    :param engine:    The current engine instance we are working with.
    :return:          Nothing.
    """
    for body in engine.bodies.values():
        body.x = x[body.offset : body.offset + len(body.x)]


def get_position_vector(engine):
    """
    This function gets the body position vector of each body and concatenates them into one global vector.

    :param engine:      The current engine instance we are working with.
    :return:
    """
    x = np.zeros((engine.number_of_nodes, 3), dtype=np.float64)
    for body in engine.bodies.values():
        x[body.offset : body.offset + len(body.x)] = body.x
    return x


def get_velocity_vector(engine):
    """
    This function retries the velocity vectors of all bodies and concatenates them into one global velocity vector.

    :param engine:      The current engine instance we are working with.
    :return:
    """
    u = np.zeros((engine.number_of_nodes, 3), dtype=np.float64)
    for body in engine.bodies.values():
        u[body.offset : body.offset + len(body.x)] = body.u
    return u


def compute_mass_matrix(engine, stats, debug_on):
    """
    Compute global mass matrix of whole system.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            CSR global mass matrix.
    """
    timer = None
    if debug_on:
        timer = Timer("compute_mass_matrix")
        timer.start()
    N = engine.number_of_nodes  # Total number of nodes in the world
    size = 0
    COO = {}
    for body in engine.bodies.values():
        row_body, col_body, data_body = ElementArrayUtil.assembly_coo(
            body.T, body.M_array
        )
        COO[body] = (row_body, col_body, data_body)
        size += len(data_body)
    row = np.zeros(size, dtype=np.int32)
    col = np.zeros(size, dtype=np.int32)
    data = np.zeros(size, dtype=np.float64)
    s = 0
    for body in engine.bodies.values():
        row_body, col_body, data_body = COO[body]
        offset = body.offset*3
        for i in range(len(row_body)):
            row[s] = row_body[i] + offset
            col[s] = col_body[i] + offset
            data[s] = data_body[i]
            s += 1
    M = sparse.csr_matrix((data, (row, col)), shape=(N * 3, N * 3))
    if debug_on:
        timer.end()
        stats["compute_mass_matrix"] = timer.elapsed
    return M


def compute_elastic_forces(x, engine, stats, debug_on):
    """
    Compute the elastic forces of whole system.

    :param x:            N-by-3 global position vector of the system.
    :param engine:       The current engine instance we are working with.
    :param stats:        A dictionary where to add more profiling and timing measurements.
    :param debug_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:             N-by-3 global elastic force vector for the system.
    """
    timer = None
    if debug_on:
        timer = Timer("compute_elastic_forces")
        timer.start()
    forces = np.zeros((engine.number_of_nodes, 3), dtype=np.float64)
    for body in engine.bodies.values():
        # Precomputed deformation gradient
        F = Native.compute_deformation_gradient(
            x[body.offset: body.offset + len(body.x)], body.T, body.invD0
        )
        # Convert soft body elastic parameters into Lame parameters
        lambda_in = MECH.first_lame(
            body.material_description.E, body.material_description.nu
        )
        mu_in = MECH.second_lame(
            body.material_description.E, body.material_description.nu
        )
        pk1_stress = body.material_description.constitutive_model.pk1_stress
        body.Fe = Native.compute_elastic_forces(
            x[body.offset: body.offset + len(body.x)],
            body.T,
            body.gradN0,
            F,
            lambda_in,
            mu_in,
            pk1_stress,
        )
        forces[body.offset : body.offset + len(body.u)] = body.Fe
    if debug_on:
        timer.end()
        stats["compute_elastic_forces"] = timer.elapsed
    return forces


def compute_traction_forces(x, engine, stats, debug_on):
    """
    Computes the traction forces applied to the whole system.

    :param x:            N-by-3 global position vector of the system.
    :param engine:       The current engine instance we are working with.
    :param stats:        A dictionary where to add more profiling and timing measurements.
    :param debug_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:             N-by-3 global traction force vector for the system.
    """
    timer = None
    if debug_on:
        timer = Timer("compute_traction_forces")
        timer.start()
    forces = np.zeros((engine.number_of_nodes, 3), dtype=np.float64)
    for body in engine.bodies.values():
        body.Ft = Native.compute_traction_forces(
            x[body.offset: body.offset + len(body.x)], body.traction_conditions
        )
        forces[
            body.offset : body.offset + len(body.u)
        ] = body.Ft
    if debug_on:
        timer.end()
        stats["compute_traction_forces"] = timer.elapsed
    return forces


def compute_damping_forces(u, engine, stats, debug_on):
    """
    Compute damping forces globally for whole system.

    :param u:            N-by-3 global velocity vector of the system.
    :param engine:       The current engine instance we are working with.
    :param stats:        A dictionary where to add more profiling and timing measurements.
    :param debug_on:     Boolean flag for toggling debug (aka profiling) info on and off.
    :return:             N-by-3 global damping force vector of the system.
    """
    timer = None
    if debug_on:
        timer = Timer("compute_damping_forces")
        timer.start()
    forces = np.zeros((engine.number_of_nodes, 3), dtype=np.float64)
    for body in engine.bodies.values():
        # TODO Kenny 2021-12-05 The problem is that we access state information stored in bodies. The high-level
        #  stepper function works differently, it extracts and assembles global vectors and matrices. Hence, those
        #  global matrices/vectors should be used. One should only access information from bodies that are constant
        #  throughout the simulation. It is a design flaw.
        body.Fd = Native.compute_damping_forces(
            body.T, body.C_array, u[body.offset : body.offset + len(body.u)]
        )
        forces[body.offset : body.offset + len(body.u)] = body.Fd
    if debug_on:
        timer.end()
        stats["compute_damping_forces"] = timer.elapsed
    return forces


def compute_external_forces(engine, stats, debug_on):
    """
    This function computes the external forces acting on all nodes in the world.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            N-by-3 global external force vector of the system.
    """
    timer = None
    if debug_on:
        timer = Timer("compute_external_forces")
        timer.start()
    forces = np.zeros((engine.number_of_nodes, 3), dtype=np.float64)
    for body in engine.bodies.values():
        body.Fext = Native.compute_gravity_forces(
            len(body.x), body.T, body.vol0, body.material_description.rho, body.gravity
        )
        forces[body.offset : body.offset + len(body.u)] = body.Fext
    if debug_on:
        timer.end()
        stats["compute_external_forces"] = timer.elapsed
    return forces


def compute_inverse_mass_matrix(engine, stats, debug_on):
    """
    This method computes the inverse mass matrix of all bodies in the world.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            The global inverse mass matrix.
    """
    timer = None
    if debug_on:
        timer = Timer("compute_inverse_mass_matrix")
        timer.start()
    M = compute_mass_matrix(engine, stats, debug_on)
    # TODO 2021-12-06 Kenny: Naively inverting the mass matrix is not smart at all. If lumped matrices are used then
    #  this should be exploited. Besides the CSR format is not the best for the sparse inverse, and further it is
    #  not exploited that the mass matrix is symmetric positive definite.
    W = sparse.linalg.inv(M)
    if debug_on:
        timer.end()
        stats["compute_inverse_mass_matrix"] = timer.elapsed
    return W


def compute_kinetic_energy(engine, stats, debug_on):
    """
    This function computes the total kinetic energy of all bodies in the world.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            The total kinetic energy of the whole system.
    """
    # 2022-03-27 Kenny TODO: Ideally we should keep consistent interfaces. This means this function should
    #                   take the velocity u-vector as input, so one specifies the state at which the energy
    #                   should be computed. Right now the method relies on values stored in the bodies.
    timer = None
    if debug_on:
        timer = Timer("compute_kinetic_energy")
        timer.start()
    kinetic = 0
    for body in engine.bodies.values():
        if body.is_fixed:
            continue
        # Below we use a FVM approach to compute energies per element. That is we use mid-point rule approximation.
        for e in range(len(body.T)):
            v = body.u[body.T[e]]  # Nodal velocities
            v_mid = (v[0] + v[1] + v[2] + v[3]) / 4.0  # Center velocity
            delta_kinetic = (
                (body.material_description.rho * body.vol0[e]) * v_mid.dot(v_mid) / 2.0
            )
            kinetic += delta_kinetic
    if debug_on:
        timer.end()
        stats["compute_kinetic_energy"] = timer.elapsed
    return kinetic


def compute_potential_energy(engine, stats, debug_on):
    """
    This function computes the total potential energy due to gravity on all bodies.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            The total potential energy of the whole system.
    """
    # 2022-03-27 Kenny TODO: Ideally we should keep consistent interfaces. This means this function should
    #                   take the position x-vector as input, so one specifies the state at which the energy
    #                   should be computed. Right now the method relies on values stored in the bodies.
    timer = None
    if debug_on:
        timer = Timer("compute_potential_energy")
        timer.start()
    potential = 0
    for body in engine.bodies.values():
        if body.is_fixed:
            continue
        # Below we use a FVM approach to compute energies per element. That is we use mid-point rule approximation.
        up_dir = - body.gravity
        for e in range(len(body.T)):
            x = body.x[body.T[e]]  # Nodal positions
            x_mid = (x[0] + x[1] + x[2] + x[3]) / 4.0  # Center position
            delta_potential = (
                body.material_description.rho * body.vol0[e]
            ) * up_dir.dot(x_mid)
            potential += delta_potential
    if debug_on:
        timer.end()
        stats["compute_potential_energy"] = timer.elapsed
    return potential


def compute_elastic_energy(engine, stats, debug_on):
    """
    This function computes the total elastic strain energy of all bodies.

    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            The total elastic energy of the whole system.
    """
    # 2022-03-27 Kenny TODO: Ideally we should keep consistent interfaces. This means this function should
    #                   take the position x-vector as input, so one specifies the state at which the energy
    #                   should be computed. Right now the method relies on values stored in the bodies.
    timer = None
    if debug_on:
        timer = Timer("compute_elastic_energy")
        timer.start()
    elastic = 0
    for body in engine.bodies.values():
        if body.is_fixed:
            continue
        # Precomputed deformation gradient
        F = Native.compute_deformation_gradient(body.x, body.T, body.invD0)
        # Convert soft body elastic parameters into Lame parameters
        lambda_in = MECH.first_lame(
            body.material_description.E, body.material_description.nu
        )
        mu_in = MECH.second_lame(
            body.material_description.E, body.material_description.nu
        )
        # Compute elastic energy of the body and accumulate it
        psi = body.material_description.constitutive_model.energy_density
        elastic += Native.compute_elastic_energy(body.vol0, F, lambda_in, mu_in, psi)

    if debug_on:
        timer.end()
        stats["compute_elastic_energy"] = timer.elapsed
    return elastic


def apply_post_stabilization(J, WJT, engine, stats, debug_on):
    """
    This function runs a simulation that computes nodal displacements in such
    a way to remove all penetrations.

    :param J:           The Contact Jacobian matrix.
    :param WJT:         The transpose Contact Jacobian, premultiplied by the inverse mass matrix.
    :param engine:      The current engine instance we are working with.
    :param stats:       A dictionary where to add more profiling and timing measurements.
    :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
    :return:            A dictionary with profiling and timing measurements.
    """
    # 2022-03-27 Kenny TODO: This method uses the contact points stored in the engine to build a right hand side
    #                   for a LCP projection type of problem. It would be nicer if instead a vector of penetration
    #                   depths is passed to the interface. So one have more explicit control over the state.
    timer = None
    if debug_on:
        timer = Timer("apply_post_stabilization")
        timer.start()
    K = len(engine.contact_points)
    g = np.zeros(3 * K, dtype=np.float64)
    for k in range(K):
        cp = engine.contact_points[k]
        if cp.g < -engine.params.min_gap_value:
            g[3 * k + 0] = cp.g

    # If the gap to correct is all zeros, then just return
    if not g.any():
        return stats

    mu = np.zeros(K, dtype=np.float64)

    sol, stats = CONTACT_SOLVERS.solve(J, WJT, g, mu, CONTACT_SOLVERS.prox_origin, engine, stats, debug_on, 
                                       prefix="post_stabilization_", 
                                       scheme=engine.params.proximal_solver)
    delta_x = WJT.dot(sol)

    # --- Convert from 3N-by-1 into N-by-3 vector format -----------------
    # TODO 2021-12-07 Kenny: This seems a bit silly to make this conversion. It should be abstracted
    #  away such that a developer can focus more on the high-level aspects of time integration and
    #  writing solver code.
    np.reshape(delta_x, (-1, 3))
    # ---------------------------------------------------------------------

    x = get_position_vector(engine)
    x += delta_x
    set_position_vector(x, engine)

    if debug_on:
        timer.end()
        stats["apply_post_stabilization"] = timer.elapsed
    return stats


def apply_boundary_conditions_on_positions(x: np.array, engine) -> None:
    """
    Apply boundary conditions.

    :param x:           The position vector to apply boundary conditions on.
    :param engine:      The current engine instance we are working with.
    """
    for body in engine.bodies.values():
        if len(body.dirichlet_conditions) > 0:
            # 2022-03-24 Kenny: TODO This is performance wise not the best
            #                    way to do it. Indices and values could be precomputed.
            indices = np.array([dc.idx for dc in body.dirichlet_conditions])
            values = np.array([dc.value for dc in body.dirichlet_conditions])
            x[indices + body.offset] = values


def apply_boundary_conditions_on_velocities(u: np.array, engine) -> None:
    """
    Apply boundary conditions.

    :param u:           The velocity vector to apply boundary conditions on.
    :param engine:      The current engine instance we are working with.
    """
    for body in engine.bodies.values():
        if len(body.dirichlet_conditions) > 0:
            # 2022-03-24 Kenny: TODO This is performance wise not the best
            #                    way to do it. Indices could be precomputed.
            indices = np.array([dc.idx for dc in body.dirichlet_conditions])
            u[indices + body.offset] = 0.0


class SemiImplicitStepper:
    """
    This class implements a semi-implicit first order Euler time-stepper.
    """
    def __init__(self, engine: Engine, debug_on: bool) -> None:
        self.log = []
        stats = {}
        self.W = compute_inverse_mass_matrix(engine, stats, debug_on)
        if debug_on:
            self.log.append(stats)

    def step(self, dt: float, engine: Engine, debug_on: bool) -> None:
        """
        This method advances time in the world modelled by the engine by the time-step size dt.

        :param dt:          The time-step size to advance the world state by.
        :param engine:      The current engine instance we are working with.
        :param debug_on:    Boolean flag for toggling debug (aka profiling) info on and off.
        :return:            None.
        """
        # 2022-03-27 Kenny TODO: This function should be redesigned to "semi_implicit_stepper" and
        #                   one should feed in the contact states, x and u vectors that it should work
        #                   on. This may it would be easier to use the stepper as a building block in
        #                   more advanced simulation loops. The semi-implicit-stepper should then be
        #                   called from API.simulate function. The stepper should not depend on state
        #                   information stored in the bodies. Bodies should just be shallow views into
        #                   global state-vectors and force vectors.
        #                   The global inverse mass and mass matrices should also only be reassembled once (or
        #                   reassembled when changes happen to the world). The current implementation rebuilds
        #                   everything in every step taken. If semi-implicit-stepper changed interface to take
        #                   the inverse-mass matrix as input then this assembly could be done elsewhere.
        timer = None
        if debug_on:
            timer = Timer("Stepper")
            timer.start()

        stats = {}

        x = get_position_vector(engine)
        apply_boundary_conditions_on_positions(x, engine)

        u = get_velocity_vector(engine)
        apply_boundary_conditions_on_velocities(u, engine)

        Fe = compute_elastic_forces(x, engine, stats, debug_on)
        Ft = compute_traction_forces(x, engine, stats, debug_on)
        Fd = compute_damping_forces(u, engine, stats, debug_on)
        # TODO 2021-12-06 Kenny: If gravity is the only external force then it is constant,
        #  and it does not make sense to recompute gravity forces in every stepper call.
        Fext = compute_external_forces(engine, stats, debug_on)
        F_tot = Fext + Fe + Ft + Fd

        # --- Convert from N-by-3 into 3N-by-1 vector format ------------------
        # TODO 2021-12-07 Kenny: This seems a bit silly to make this conversion. It should be abstracted
        #  away such that a developer can focus more on the high-level aspects of time integration and
        #  writing solver code.
        u = np.reshape(u, (-1))
        F_tot = np.reshape(F_tot, (-1))
        # ---------------------------------------------------------------------

        u_prime = u + dt * self.W.dot(F_tot)

        # TODO 2021-12-07 Kenny: The whole collision detection system uses body positions (state info) stored in body
        #  instances and not stored in global position vectors. This provides a design problem if one wish to perform
        #  collision detection at a future predicted position. Currently, in our semi-implicit first order
        #  time-integration this does not prove to be a problem as we do collision detection at the current/initial
        #  positions only. However, implementing backward Euler type of methods would require us to do collision
        #  detection at a future predicted position. Solution would be to pass the global x-vector to the collision
        #  detection sub-system.

        # Find contact points
        stats = CD.run_collision_detection(engine, stats, debug_on)

        # Calculate contact forces
        J = None
        WJT = None
        WPc = np.zeros(u_prime.shape, dtype=np.float64)
        if len(engine.contact_points) > 0:

            J = compute_jacobian_matrix(engine, stats, debug_on)
            WJT = self.W.dot(J.T)

            if engine.params.use_pre_stabilization:
                raise ValueError("Pre-stabilization has not been implemented")

            mu = get_friction_coefficient_vector(engine)
            b = J.dot(u_prime)
            sol, stats = CONTACT_SOLVERS.solve(J, WJT, b, mu, CONTACT_SOLVERS.prox_sphere, engine, stats, debug_on, 
                                               prefix="", 
                                               scheme=engine.params.proximal_solver)
            WPc = WJT.dot(sol)

        # --- Convert from 3N-by-1 into N-by-3 vector format -----------------
        # TODO 2021-12-07 Kenny: This seems a bit silly to make this conversion. It should be abstracted
        #  away such that a developer can focus more on the high-level aspects of time integration and
        #  writing solver code.
        u_prime = np.reshape(u_prime, (-1, 3))
        WPc = np.reshape(WPc, (-1, 3))
        # ---------------------------------------------------------------------

        # Semi-implicit time integration
        apply_boundary_conditions_on_velocities(u_prime, engine)
        u = u_prime + WPc
        apply_boundary_conditions_on_velocities(u, engine)
        x += u * dt
        apply_boundary_conditions_on_positions(x, engine)

        set_velocity_vector(u, engine)
        set_position_vector(x, engine)

        if engine.params.use_post_stabilization:
            if len(engine.contact_points) > 0:
                # TODO 2021-12-07 Kenny: Here we re-use the contact Jacobian information we retried from the collision
                #  detection query earlier on. Again the interface here might not be generic enough for fitting into
                #  other time-stepping schemes. One could "build" the post stabilization routine as a specialized
                #  stepper in its own right. This might be cool for pre-processing of simulations to make sure
                #  no penetrations are initially present.
                stats = apply_post_stabilization(J, WJT, engine, stats, debug_on)
        
        # Update time stamp
        engine.params.time_stamp += 1

        if debug_on:
            timer.end()
            stats["stepper_time"] = timer.elapsed
            stats["dt"] = dt
            stats["contact_points"] = len(engine.contact_points)
            stats["kinetic_energy"] = compute_kinetic_energy(engine, stats, debug_on)
            stats["potential_energy"] = compute_potential_energy(engine, stats, debug_on)
            stats["elastic_energy"] = compute_elastic_energy(engine, stats, debug_on)
            stats["max_penetration"] = get_largest_penetration_error(engine)
            self.log.append(stats)
