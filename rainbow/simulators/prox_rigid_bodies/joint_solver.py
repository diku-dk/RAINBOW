






def compute_joint_jacobian_matrix(engine):
    """
    This function creates the contact Jacobian matrix by iterating over all contact points in the
    engine and assembling the Jacobian.

    :param engine:   A reference to the engine holding the contact point information.
    :return:         The Jacobian matrix.
    """
    N = len(engine.bodies)
    K = len(engine.hin)

    cols = N * 6
    rows = K * 4

    data = np.zeros(K * 48, dtype=np.float64)
    row = np.zeros(K * 48, dtype=np.float64)
    col = np.zeros(K * 48, dtype=np.float64)
    next_entry = 0
    for k in range(K):
        cp = engine.contact_points[k]
        # Compute the vector arms from point of contact to center of mass of the rigid bodies.
        rA = cp.p - cp.bodyA.r
        rB = cp.p - cp.bodyB.r
        # Compute a contact frame where two first vectors span the contact plane, and the last
        # vector is orthogonal and defines the contact normal direction of the local contact frame system.
        vs, vt, vn = V3.make_orthonormal_vectors(cp.n)
        # Now compute the Jacobian matrix blocks, the essentially maps the body velocities into
        # a relative contact space velocity.
        JA_v = -np.array([vn, vs, vt, V3.zero()], dtype=np.float64)
        JA_w = -np.array(
            [V3.cross(rA, vn), V3.cross(rA, vs), V3.cross(rA, vt), vn], dtype=np.float64
        )
        JB_v = np.array([vn, vs, vt, V3.zero()], dtype=np.float64)
        JB_w = np.array(
            [V3.cross(rB, vn), V3.cross(rB, vs), V3.cross(rB, vt), vn], dtype=np.float64
        )
        # Now we have all the values we need. Next step is to fill the values into the global Jacobian
        # matrix in the right "entries". The for loops below is doing all the index bookkeeping for making
        # this mapping.
        idx_A = cp.bodyA.idx
        idx_B = cp.bodyB.idx
        for i in range(4):
            row_idx = k * 4 + i
            col_idx = idx_A * 6
            for j in range(3):
                data[next_entry] = JA_v[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx += 3
            for j in range(3):
                data[next_entry] = JA_w[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx = idx_B * 6
            for j in range(3):
                data[next_entry] = JB_v[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
            col_idx += 3
            for j in range(3):
                data[next_entry] = JB_w[i, j]
                row[next_entry] = row_idx
                col[next_entry] = col_idx + j
                next_entry = next_entry + 1
    # Finally, we can create the specific sparse matrix format that is suitable for our
    # simulator. Here, we go with CSR, because we in most places just need efficient
    # matrix-vector operations.
    J = sparse.csr_matrix((data, (row, col)), shape=(rows, cols))
    return J

