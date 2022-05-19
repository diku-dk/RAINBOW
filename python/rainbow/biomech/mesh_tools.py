import rainbow.geometry.volume_mesh_wildmesh as WM
import rainbow.geometry.volume_mesh as VM
import rainbow.geometry.surface_mesh as SURF
import numpy as np
import igl

### Miscellaneous mesh procedures


def _clean_mesh(V, S, d=500):
    """
    :param V: #V x 3 array of vertices
    :param S: #S x 3/4 array of faces/tetrahedra
    :return V: #V x 3 array of vertices without duplicated
               and unreferenced vertices
    :return S: #S x 3/4 array of faces/tetrahedra without duplicates
    :remarks:
        Epsilon is 1/d'th of the smallest edge in (V, S).
        Any vertices closer than epsilon are merged.
    """
    eps = np.min(igl.edge_lengths(V, S)) / d
    V, S = igl.remove_duplicate_vertices(V, S, eps)[0:4:3]
    if S.shape[1] == 3:
        S = igl.resolve_duplicated_faces(S)[0]
    V, S = igl.remove_unreferenced(V, S)[0:2]
    return V, S


def _stack_meshes(Va, Sa, Vb, Sb, La=None, Lb=None):
    """
    :param Va: #Va x 3 array of vertices
    :param Sa: #Sa x 3/4 array of faces/tetrahedra
    :param La: #Sa x 1 array of simplicial labels
    :param Vb: #Vb x 3 array of vertices
    :param Sb: #Sb x 3/4 array of faces/tetrahedra
    :param Lb: #Sb x 1 array of simplicial labels
    :return V: #Va + #Vb x 3 array of vertices 
    :return S: #Sa + #Sb x 3/4 array of facces/tetrahedra
    :return L: #Sa + #Sb x 1 array of simplicial labels
    :remarks:
        Stacks two meshes into one. Mesh a on top.
    """
    if La is None:
        V = np.concatenate((Va, Vb))
        S = np.concatenate((Sa, Sb + len(Va)))
        return V, S
    else:
        V = np.concatenate((Va, Vb))
        S = np.concatenate((Sa, Sb + len(Va)))
        L = np.concatenate((La, Lb))
        return V, S, L


def _flip_orientation(S):
    """
    :param S: #S x 3/4 array of faces/tetrahedra
    :return Sc: #S x 3/4 array faces/tetrahedra with flipped orientation w.r.t. S
    """
    Sc = np.copy(S)
    # Flip orientation by interchanging two points
    Sc[:, [0, 1]] = Sc[:, [1, 0]]
    return Sc


def _remove_ears(V, S):
    """
    :V: N x 3 array of vertices
    :S: M x 3/4 array of faces/tetrahedra
    :returns: Mesh defined by (V, S) with ears removed. A tet ear is
              a tet with three free faces.
    """
    if not (S.shape[1] == 3 or S.shape[1] == 4):
        raise Exception(f"Expected simplices of dimension 3 or 4 but got {S.shape[1]}")
    if S.shape[1] == 3:
        I_ears = igl.ears(S)[0]
    if S.shape[1] == 4:
        TT = igl.tet_tet_adjacency(S)[0]
        I_ears = np.where(np.sum((TT < 0), axis=1) == 3)[0]
    S_new = np.delete(S, I_ears, axis=0)
    V, S = igl.remove_unreferenced(V, S_new)[0:2]
    return V, S


def _keep_removing_ears(V, S):
    """
    :V: N x 3 array of vertices
    :S: M x 3/4 array of faces/tetrahedra
    :returns: Mesh defined by (V, S) with ears iteratively removed, i.e. 
              if the removal of an ear results in one or more additional
              ears, these will be removed in the next iteration, until no
              ears are present. A tet ear isa tet with three free faces.
    """
    cur_elms = S.shape[0]
    while True:
        V, S = remove_ears(V, S)
        if S.shape[0] == cur_elms:
            break
        else:
            cur_elms = S.shape[0]
    return V, S


### Mesh verification procedures


def verify_mesh(V, S, verbose=True):
    """
    :param V: N x 3 array of vertices
    :param S: N x 3/4 array of simplices (faces/tetrahedra)
    :return FLAG: True if the mesh (V, S) passes verification. False otherwise.
    """
    # Check for correct dimensionality
    if not (S.shape[1] == 3 or S.shape[1] == 4):
        raise Exception(f"Expected simplices of dimension 3 or 4 but got {S.shape[1]}")

    FLAG = True

    # Perform verification sub-routines

    if verbose:
        print("Starting verification of element orientation")

    if not _verify_orientation(V, S):
        if S.shape[1] == 3:
            print("    Inconsistent orientation found")
        if S.shape[1] == 4:
            print("    Negative orientation found")
        FLAG = False

    if verbose:
        print("Starting verification of mesh features")

    if not _verify_unique_simplices(V, S):
        print("    Found duplicated simplices")
        FLAG = False

    if not _verify_no_duplicated_vertices(V, S):
        print("    Found duplicated vertices")
        FLAG = False

    if not _verify_no_unreferenced_vertices(V, S):
        print("    Found unreferenced vertices")
        FLAG = False

    if verbose:
        print("Starting verification of mesh manifoldness")

    if not _verify_manifoldness(V, S):
        print("    Found non-manifold element")
        FLAG = False

    if verbose:
        print("Starting verification of number of mesh components")

    if not _verify_one_component(S):
        print("    Mesh contains more than one component")
        FLAG = False

    if verbose:
        print("Starting verification of element connectivity")

    if not _verify_interior(V, S):
        print("    An interior link is referenced more than twice")
        FLAG = False

    return FLAG


def _verify_triangle_orientation(F):
    """
    :F: N x 3 array of faces
    :returns: True if all triangles are oriented consistently
              and otherwise false
    """
    CONSISTENT_ORIENTATION = False
    edges = igl.edges(F)
    for e in edges:
        # Get indices of faces referencing edge e
        idx = np.where(np.sum(np.isin(F, e), axis=1) == 2)[0]

        # Not closed surface but continue to next edge anyway
        if len(idx) != 2:
            continue

        # Append first element to each face for easy checking
        F1 = np.append(F[idx[0]], F[idx[0]][0])
        F2 = np.append(F[idx[1]], F[idx[1]][0])

        # Check if orientation of e is consistent with e in F1
        in_F1 = False
        for i in range(3):
            if F1[i] == e[0] and F1[i + 1] == e[1]:
                in_F1 = True

        # If orientation of e is consistent with e in F1 then
        # e in F2 must be of opposite orientation
        if in_F1:
            for i in range(3):
                if F2[i] == e[1] and F2[i + 1] == e[0]:
                    CONSISTENT_ORIENTATION = True

        # If orientation of e is inconsistent with e in F1 then
        # e must have same orientation as e in F2
        else:
            for i in range(3):
                if F2[i] == e[0] and F2[i + 1] == e[1]:
                    CONSISTENT_ORIENTATION = True

        # If orientation is not consistent return false
        if not CONSISTENT_ORIENTATION:
            return False

        # Reset for next edge
        CONSISTENT_ORIENTATION = False

    # No inconsistent orientations found so return true
    return True


def _verify_orientation(V, S):
    """
    :param V: N x 3 array of vertices
    :param S: N x 3/4 array of simplices (faces/tetrahedra)
    :return: If tet mesh returns true if the mesh has no only positive orientation. 
             False otherwise.
             If triangle mesh returns true if faces are consistently oriented.
             False otherwise.  
    """
    if S.shape[1] == 4:
        # Create matrix of cross products
        C = np.cross(V[S][:, 1] - V[S][:, 0], V[S][:, 2] - V[S][:, 0])
        # Create matrix with rows to dot each cross product with
        L = V[S][:, 3] - V[S][:, 0]
        # Compute dot product and thereby the orientations
        O = np.sum(C * L, axis=1)
        # All must be positive
        return np.all(O > 0)
    elif S.shape[1] == 3:
        return _verify_triangle_orientation(S)
    else:
        raise Exception(f"Expected simplices of dimension 3 or 4 but got {S.shape[1]}")


def _verify_unique_simplices(V, S):
    """
    :param V: N x 3 array of vertices
    :param S: N x 3/4 array of simplices (faces/tetrahedra)
    :return: True if the mesh (V, S) has only unique simplices.
    """
    NS = len(S)  # Number of simplices in S
    US = len(igl.unique_simplices(S)[0])  # Number of unique simplices in S
    return NS == US


def _verify_no_unreferenced_vertices(V, S):
    """
    :param V: N x 3 array of vertices
    :param S: N x 3/4 array of simplices (faces/tetrahedra)
    :return: True if the mesh (V, S) has no unreferenced vertices.
    """
    NV = len(V)  # Number of vertices in V
    UR = len(igl.remove_unreferenced(V, S)[0])  # Number of referenced vertices
    return NV == UR


def _verify_no_duplicated_vertices(V, S, eps=1e-16):
    """
    :param V: N x 3 array of vertices
    :param S: N x 3/4 array of simplices (faces/tetrahedra)
    :return: True if the mesh (V, S) has no duplicated vertices.
    """
    NV = len(V)  # Number of vertices in V
    UV = len(igl.remove_duplicates(V, S, eps)[0])  # Number of unique vertices in V
    return NV == UV


def _verify_one_component(S):
    """
    :param S: N x 3/4 array of simplices (faces/tetrahedra)
    :return: True if S is a geometry consisting of a single component. False otherwise.
    """
    return igl.connected_components(igl.adjacency_matrix(S))[0] == 1


def _verify_interior(V, S):
    """
    :param S: N x 3/4 array of simplices (faces/tetrahedra)
    :return: True if interior faces/edges has two tets/triangles. False otherwise.
    """
    # Compute dimensionality of boundary elements
    dim = S.shape[1] - 1

    # If surface mesh
    if S.shape[1] == 3:
        # Assert number of faces adjacent to all edges is two (we want closed surface meshes)
        unique_num_edge_references = np.unique(
            np.unique(igl.edge_topology(V, S)[1], return_counts=True)[1]
        )
        if (
            unique_num_edge_references[0] == dim
            and len(unique_num_edge_references) == 1
        ):
            return True
        else:
            return False

    # If volume mesh
    # TODO: Come up with way to improve performance, i.e. parallelize using numpy
    for [i, j, k, m] in S:
        L = [[i, j, k], [i, j, m], [j, k, m], [k, i, m]]

        # Loop through each link (edge/face)
        for l in L:
            # Check that each interior link is being referenced by one or two simplices
            if len(np.where(np.sum(np.isin(S, l), axis=1) == dim)[0]) > 2:
                return False
    return True


def _compute_euler_characteristic(F):
    """
    :param F: N x 3 array of faces.
    :return: The Euler characteristic of the mesh with faces F.
    """
    N_v = len(np.unique(F))
    N_e = len(igl.edges(F))
    N_f = len(F)
    return N_v - N_e + N_f


def _verify_manifoldness(V, S):
    """
    :param S: N x 3/4 array of simplices (faces/tetrahedra)
    :return: True if mesh (V, S) is a manifold. False otherwise.
    """
    for i in range(len(V)):
        # Indices into S of faces/tetrahedra referencing vertex i.
        indices = np.nonzero(np.isin(S, i))[0]
        # Unreferenced vertex so just continue
        if len(indices) == 0:
            continue

        # Surface mesh
        if S.shape[1] == 3:
            # Find all edges of faces referencing vertex i
            e = igl.edges(S[indices])
            # Select all edges not referencing vertex i, i.e. boundary edges
            boundary_indices = np.where(np.sum(np.isin(e, i), axis=1) == 0)[0]
            # Try to compute a path along the boundary edges, i.e. the simplicial link
            boundary_path = igl.edges_to_path(e[boundary_indices])[0]
            # Check that simplicial link is a loop
            if boundary_path[0] != boundary_path[-1]:
                return False

        # Volume mesh
        if S.shape[1] == 4:
            # Check for manifoldness via the Euler characteristic of the simplicial link
            if _compute_euler_characteristic(igl.boundary_facets(S[indices])) != 2:
                return False

    return True


### Naive mesh snapping procedure


def snap_meshes(Va, Ta, Vb, Tb, La=None, Lb=None, eps=1e-16):
    """
    :param Va: #Va x 3 array of vertices
    :param Ta: #Ta x 4 array of tetrahedra
    :param Vb: #Vb x 3 array of vertices
    :param Tb: #Tb x 4 array of tetrahedra
    :param La: #Ta x 1 array of simplicial labels
    :param Lb: #Tb x 1 array of simplicial labels
    :param eps: Distance determining if vertices should be
                considered duplicates
    :return V: #V x 3 array of vertices
    :return T: #T x 4 array of tetrahedra
    :return L: #T x 1 array of tetrahedra labels
    :remarks:
        (V, T) is the mesh resulting from snapping mesh b to mesh a, i.e. any 
        vertices in mesh b closer then eps is snapped to the surface of mesh a.
        There maybe penetrations and gaps between the faces, but vertices from  
        mesh b should recide on the faces of mesh a.
    """
    # Find distances and closest points on (Va, Fa)
    S, _, C = igl.signed_distance(Vb, Va, Ta)
    # Compute indices where the distance from Vb are within epsilon band
    I = np.where((-eps < S) & (S < eps))[0]
    # Move close vertices from mesh b to closest point on mesh a
    Vb[I] = C[I]
    # Stack the two meshes and label vectors
    V, T, L = _stack_meshes(Va, Ta, Vb, Tb, La=La, Lb=Lb)
    # Remove duplicated vertices (the ones we just put at the same spot)
    # This should >not< change the ordering of tets which would destroy the labeling
    V, T = igl.remove_duplicate_vertices(V, T, eps)[0:4:3]
    return V, T, L
