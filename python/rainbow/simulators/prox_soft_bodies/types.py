import rainbow.math.vector3 as V3
from rainbow.simulators.prox_soft_bodies.mechanics import *
import rainbow.geometry.spatial_hashing as SH


class SurfacesInteraction:
    """
    This class contains parameters that describe the physical
    interaction between two types of material.
    
    Such as coefficient of friction.
    """

    def __init__(self):
        self.mu = V3.ones()  # Coefficients of Friction


class SurfacesInteractionLibrary:
    """
    This class keeps track of all the different combinations of
    types of surface material interactions we have created/specified.
    """

    def __init__(self):
        self.storage = dict()
        self.storage[
            ("default", "default")
        ] = SurfacesInteraction()  # Default surface interaction.

    def get_interaction(self, A, B):
        """
        This method retrieves the surface interaction description
        between the given material names.
        """
        key = (A, B) if A < B else (B, A)
        if key in self.storage:
            return self.storage[key]
        return None

    def exist_interaction(self, A, B):
        """
        This method tests if a surface interaction has been described
        between the two materials given by the names A and B.
        """
        key = (A, B) if A < B else (B, A)
        if key in self.storage:
            return True
        return False

    def exist_material(self, name):
        """
        This method tests if the given material name exist in the
        current storage.
        """
        for key in self.storage:
            if name in key:
                return True
        return False


class ContactPoint:
    """
    This is a contact point data type.
    """

    def __init__(
        self, bodyA, bodyB, idx_tetA, idx_tetB, omegaA, omegaB, position, normal, gap
    ):
        """
        A contact point is coupled to a tetrahedron from body A and a tetrahedron from body B. The point can
        hence be expressed in terms of barycentric coordinates wrt the tetrahedrons (4 numbers from each). The
        tetrahedral indices and barycentric coordinates weight and indices are needed for creating the
        contact Jacobian.

        :param bodyA:       A reference to body A.
        :param bodyB:       A reference to body B.
        :param idx_tetA:    The index of tetrahedron from body A.
        :param idx_tetB:    The index of tetrahedron from body B.
        :param omegaA:      The barycentric coordinates of the position with respect to the tetrahedron of body A
        :param omegaB:      The barycentric coordinates of the position with respect to the tetrahedron of body B
        :param position:    A 3D world space position that specifies the contact point location.
        :param normal:      A 3D world space normal vector. By convention the normal always point from body A
                            towards body B.
        :param gap:         The penetration depth measure or gap that measures either contact separation or penetration.
        """
        if abs(1.0 - V3.norm(normal)) > 0.1:
            raise RuntimeError(
                "ContactPoint.init() was called with non-unit size normal"
            )
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.idx_tetA = idx_tetA
        self.idx_tetB = idx_tetB
        self.omegaA = np.copy(omegaA)
        self.omegaB = np.copy(omegaB)
        self.p = np.copy(position)
        self.n = np.copy(normal)
        self.g = gap


class DirichletCondition:
    """
    This data type represent a single Dirichlet condition on one variable.
    """

    def __init__(self, idx, value):
        """
        This initializer creates an instance that describes a nodal Dirichlet boundary condition.

        :param idx:       The nodal index on where to apply the Dirichlet Condition
        :param value:     The value to enforce, i.e. x[idx] = value for all times.
        """
        self.idx = idx
        self.value = np.copy(value)


class TractionCondition:
    """
    This data type represent a constant traction on a triangle, that is a constant load on the triangle.
    """

    def __init__(self, i, j, k, load):
        """
        This initializer creates an instance the keep relevant information for computing the constant per triangle
        traction forces applied to a single surface triangle. The surface triangle is represented by the nodal
        indices and the load is specified by a loading vector.

        :param i:     The nodal global index of first vertex in the surface triangle where traction is applied.
        :param j:     The nodal global index of second vertex in the surface triangle where traction is applied.
        :param k:     The nodal global index of third vertex in the surface triangle where traction is applied.
        :param load:  The load vector to be applied to the triangle.  The load is considered to be a constant vector in
                      the world.
        """
        self.i = i
        self.j = j
        self.k = k
        self.traction = np.copy(load)


class MaterialDescription:
    """
    A material description consists of all parameter that describes in internal/bulk of the material.
    """

    def __init__(self):
        self.constitutive_model = COR  # The constitutive model used for this material.
        self.c = 0  # The damping coefficient
        self.rho = 1000  # The mass density
        self.E = 10 ** 5  # Young modulus
        self.nu = 0.3  # Poisson ratio
        self.name = "default"  # The name of material that this soft body is made of (used to find friction)


class SoftBody:
    """
    This class contains all data needed for a single soft body.
    """

    def __init__(self, name):
        """
        Initializing a soft body. This creates a default instance of a soft body without any mesh or materials
        etc. associated with it.

        :param name:     An unique name for the soft body.
        """
        # We assume we have N vertices, K tetrahedrons and H surface triangles
        self.name = name
        self.idx = None  # Unique index of soft body, used to access body information stored in arrays.
        self.offset = 0  # Starting global index used to access body information stored in arrays
        self.T = None  # Array of tetrahedral elements (K-by-4 array).
        self.surface = None  # Array of triangle surface elements (H-by-3 array).
        self.owners = None  # Surface triangle owner information, owners[triangle] -> (tetrahedron, opposing vertex)
        self.neighbors = None  # Tetrahedron neighbor information
        self.bvh = None  # A kDOP bvh hierarchy of surface elements.
        self.grid = None  # A signed distance field of the surface mesh.
        self.dirichlet_conditions = []  # List of nodal dirichlet conditions to apply to the object.
        self.traction_conditions = []  # List of surface traction conditions to apply to the object.
        self.x0 = None  # The material/un-deformed vertex coordinates of the soft body (N-by-3 array).
        self.invD0 = None  # Precomputed inverted material edge-vector matrices (K 3-by-3 array).
        self.vol0 = None  # The material volume of each tetrahedron.
        self.gradN0 = None  # Precomputed material space face normals, or the negative gradient of the shape function.
        self.M_array = None  # Precomputed mass element array.
        self.C_array = None  # Precomputed damping element array.
        self.x = None  # The spatial (deformed) vertex coordinates of the soft body (N-by-3 array).
        self.u = None  # The spatial velocity of each vertex.  This is an N-by-3 array.
        self.Fe = None  # The spatial elastic nodal forces. This is an N-by-3 array.
        self.Ft = None  # The traction nodal forces. This is an N-by-3 array.
        self.Fd = None  # The damping nodal forces. This is an N-by-3 array.
        self.Fext = None  # The external nodal forces. This is an N-by-3 array.
        self.material_description = None  # The material this soft body is made of.
        self.gravity = V3.make(0, -9.82, 0)  # Gravitational acceleration on this body.
        self.is_lumped = True  # Boolean flag that indicates if the body should be lumped or not.
        self.is_fixed = False  # Boolean flag that indicates if tbe body should be fixed or not.


class Parameters:
    """
    This call contains all parameter values that control the simulator.
    """

    def __init__(self):
        """
        This initializes a default instance of our parameter class. The parameter class controls the behavior of all
        numerical methods and solver specific settings as well as toggling on/off different modeling terms. It does
        not describe any material/mesh like settings. Those are defined individually for each soft body.
        """
        self.time_step = (
            0.001  # The desired time step to use when taking one simulation solver step.
        )
        self.max_iterations = 200  # Maximum number of Gauss-Seidel iterations.
        self.use_pre_stabilization = (
            False  # Turning pre-stabilization on and off for correcting errors.
        )
        self.use_post_stabilization = (
            False  # Turning post-stabilization on and off for correcting errors.
        )
        self.gap_reduction = (
            0.5  # The amount of gap (=penetration) to reduce during stabilization.
        )
        self.min_gap_value = (
            0.001  # The minimum gap (=penetration) for correcting errors.
        )
        self.max_gap_value = (
            0.01  # The maximum gap (=penetration) for correcting errors.
        )
        self.absolute_tolerance = 0.001  # The absolute tolerance value.
        self.relative_tolerance = 0.0001  # The relative tolerance value.
        self.ellipsoid_max_iterations = (
            100  # The maximum number of iterations in the prox ellipsoid binary search.
        )
        self.ellipsoid_expansion = 1.5  # The scalar expansion coefficient of the prox ellipsoid binary search interval.
        self.ellipsoid_tolerance = (
            10e-10  # The tolerance for the prox ellipsoid binary search.
        )
        self.nu_reduce = (
            0.7  # How big a factor to reduce r by if divergence is detected.
        )
        self.nu_increase = (
            1.3  # How big a factor to increase r by if convergence is detected.
        )
        self.too_small_merit_change = (
            0.01  # The smallest a merit change is allowed before r factor is increased.
        )
        self.contact_optimization_max_iterations = (
            8  # The maximum number of iterations for contact optimization.
        )
        self.contact_optimization_tolerance = (
            0  # The tolerance for the frank wolfe collision detection algorithm.
        )
        self.bvh_chunk_size = 255  # Number of nodes for a K-DOP bvh subtree, a chunk.
        self.K = 3  # The number of directions to used in the k-DOP bounding volumes.
        self.envelope = (
            0.1  # Any geometry within this distance generates a contact point.
        )
        self.resolution = 64  # The number of grid cells along each axis in the signed distance fields.
        self.proximal_solver = "gauss_seidel" # or "gauss_seidel", "parallel_gauss_seidel", "parallel_jacobi", "parallel_jacboi_hybrid"
        self.use_spatial_hashing = True  # Boolean flag that indicates if spatial hashing should be used instead of the BVH or not.
        self.time_stamp = 0 # The time step to use when simulating forward.

class Engine:
    """
    This class holds all data that describes the world that we are currently simulating.
    """
    # 2022-03-27 Kenny TODO: The word engine may be a little misplaced. Perhaps we should rename this to
    #                   "World" and then keep Parameters as a separate argument for our interfaces? This would create
    #                   a nicer API interface too. Like API.create_world and API.create_body(..,world,..)
    def __init__(self):
        """
        This initializes an engine instance which holds all the configuration data of the world that is to
        be simulated.
        """
        self.simulator_type = 'soft_body' # simulation type for the engine
        self.materials = dict()  # All materials that exist in the world.
        self.materials[
            "default"
        ] = (
            MaterialDescription()
        )  # Default material to use for bodies if none is specified.
        self.surfaces_interactions = (
            SurfacesInteractionLibrary()
        )  # Interactions between different type of surfaces.
        self.params = (
            Parameters()
        )  # All input parameters to control behavior of engine/solver etc.
        self.bodies = dict()  # All bodies that exist in the world
        self.contact_points = (
            []
        )  # All contact points in last call of collision detection system.
        self.number_of_nodes = 0  # The total number of nodes in the world.
        self.stepper = None  # A reference to the time-stepper used to simulator forward.
        self.hash_grid = SH.HashGird()
        
