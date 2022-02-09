import numpy as np
import json
import wildmeshing as wm

# import wildmeshing.wildmeshing as wm # From cpp/external/wildmeshing-python


class Options:
    def __init__(self):
        """
        Read more about options for tetwild here https://wildmeshing.github.io/python/
        """
        self.stop_quality = (
            10.0  # "Specify max AMIPS energy for stopping mesh optimization"
        )
        self.max_its = 80  # "Max number of mesh optimization iterations"
        self.stage = 2  # "Specify envelope stage"
        self.stop_p = -1  #
        self.epsilon = 1e-3  # "relative envelope epsilon_r. Absolute epsilonn = epsilon_r * diagonal_of_bbox"
        self.edge_length_r = (
            1.0 / 20.0
        )  # "Relative target edge length l_r. Absolute l = l_r * diagonal_of_bbox"
        self.skip_simplify = False
        self.smooth_open_boundary = False
        self.manifold_surface = False
        self.correct_surface_orientation = False
        self.all_mesh = False
        # TODO 2021-03-25 Kenny: Does not seem to be used by wildmeshing.Tetrahedralizer?
        self.coarsen = True
        # TODO 2021-03-25 Kenny: Add a saving option to interface like we have in tetgen Options


def create_mesh(V, F, options, VB=None, TB=None, AB=None):
    """
    Make tetrahedral mesh from input surface mesh.

    :param V: This  is N-by-3 numpy array of X, Y and Z coordinates of the input surface mesh vertices. Number of
    rows is equal to number of vertices.

    :param F: This is K-by-3 input array of triangle faces from the input surface mesh. Each row corresponds to a
    triangle and the columns hold the indices of the vertices.

    :param options: This is an instance of the Options class containing parameter values that should be used.


    :param VB: In case a back-ground mesh is used to control the mesh element sizes then this input parameter holds the
     X,Y, Z coordinates of the vertices. If no background mesh is needed then argument is set to None.

    :param TB: In case a background mesh is used this argument holds the tetrahedral elements of it. This is a K-by-4
    input array where K is the total number of elements.

    :param AB: In case a background mesh is used this argument holds the vertex attribute values of this mesh. That
    is it has the same number of rows as the VB-argument, but only one-column. The attribute value at a vertex can
    conceptually be thought of as a sample point of a sizing function.

    :return: A tuple of output vertex array (N-by-3) of X,Y and Z coordinates and output tetrahedral element matrix (
    K-by-4) with vertex indices of each element, where N is the number of vertices and K is the number of tetrahedral.
    """
    tetra = wm.Tetrahedralizer(
        stop_quality=options.stop_quality,
        max_its=options.max_its,
        stage=options.stage,
        stop_p=options.stop_p,
        epsilon=options.epsilon,
        edge_length_r=options.edge_length_r,
        skip_simplify=options.skip_simplify,
        # coarsen=options.coarsen
    )
    tetra.set_mesh(V, F)

    if not (VB is None and TB is None and AB is None):
        tetra.set_sizing_field(VB, TB, AB)

    tetra.tetrahedralize()
    mesh = tetra.get_tet_mesh(
        smooth_open_boundary=options.smooth_open_boundary,
        manifold_surface=options.manifold_surface,
        correct_surface_orientation=options.correct_surface_orientation,
        all_mesh=options.all_mesh,
    )
    return mesh[0], mesh[1]


class Arg:
    def __init__(self, number):
        self.id = number

    def json(self):
        return str(self.id)


class Union:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def json(self):
        data = {"operation": "union", "left": self.left, "right": self.right}
        return data


class Difference:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def json(self):
        data = {"operation": "difference", "left": self.left, "right": self.right}
        return data


class Intersection:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def json(self):
        data = {"operation": "intersection", "left": self.left, "right": self.right}
        return data


def _make_json_string(x):
    try:
        x1 = x.json()["operation"]
        x2 = _make_json_string(x.json()["left"])
        x3 = _make_json_string(x.json()["right"])
        return '{"operation": "' + x1 + '", "left": ' + x2 + ', "right": ' + x3 + "}"
    except:
        return x.json()


def create_mesh_from_csg(Vs, Fs, CSG, options):
    """

    :param Vs:
    :param Fs:
    :param CSG:
    :param options:
    :return:
    """
    tetra = wm.Tetrahedralizer(
        stop_quality=options.stop_quality,
        max_its=options.max_its,
        stage=options.stage,
        stop_p=options.stop_p,
        epsilon=options.epsilon,
        edge_length_r=options.edge_length_r,
        skip_simplify=options.skip_simplify,
        # coarsen=options.coarsen
    )
    tetra.set_meshes(Vs, Fs)
    tetra.tetrahedralize()

    mesh = tetra.get_tet_mesh_from_csg(_make_json_string(CSG))

    return mesh[0], mesh[1]
