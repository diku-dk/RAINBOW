import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q


class SurfacesInteraction:
    """
    This class contains parameters that describe the physical
    interaction between two types of material.

    Such as coefficient of friction.
    """

    def __init__(self):
        self.mu = V3.ones()  # Coefficients of Friction
        self.epsilon = 0.0  # Coefficient of restitution


class SurfacesInteractionLibrary:
    """
    This class keeps track of all the different combinations of
    types of surface material interactions we have created/specified.
    """

    def __init__(self):
        self.storage = dict()
        self.storage[("default", "default")] = SurfacesInteraction()

    def get_interaction(self, A, B):
        """
        Retrieve surface interaction between a pair of materials.

        :param A:   Name of first material.
        :param B:   Name of second material.
        :return:    The interaction parameters between materials A and B or default interaction if it does not exist.
        """
        key = (A, B) if A < B else (B, A)
        if key in self.storage:
            return self.storage[key]
        return self.storage[("default", "default")]

    def exist_interaction(self, A, B):
        """
        Test if an interaction instance exist between the two given materials.

        :param A:   Name of first material.
        :param B:   Name of second material.
        :return:    Boolean value indicating if the interaction exist.
        """
        key = (A, B) if A < B else (B, A)
        if key in self.storage:
            return True
        return False

    def exist_material(self, name):
        """
        Test if a given material exist in the interaction library. Meaning that some
        interactions pair exist with that material name.

        :param name:   The name of the material.
        :return:       boolean value indicating if the material name was encountered in any interactions.
        """
        for key in self.storage:
            if name in key:
                return True
        return False


class ForceCalculator:
    """
    Base class for all force types that can be applied to a rigid body.
    """

    def __init__(self, force_type, name):
        """
        Create a force instance.

        :param force_type: A unique string telling what kind of force type this is.
        :param name:       A unique name identifying this instance of a "force".
        """
        self.force_type = force_type
        self.name = name


class Gravity(ForceCalculator):
    """
    This class defines gravity force type.
    """

    def __init__(self, name):
        """
        Create a gravity force instance.

        :param name:   A unique force name among all types of forces.
        """
        super().__init__("Gravity", name)
        self.g = 9.81  # Acceleration of gravity
        self.up = V3.j()  # Up direction

    def compute(self, body, r, q, v, w):
        """
        Compute the effect of the force type acting on a rigid body.

        :param body:    The reference to the rigid body that the force is acting on. Often used to access
                        information about mass or other properties of the rigid body.
        :param r:       The position of the center of mass of the rigid body.
        :param q:       The orientation of the body frame of the rigid body.
        :param v:       The linear velocity of the center of mass of the rigid body.
        :param w:       The angular velocity of the rigid body.
        :return:        A pair of 3D vectors representing the force and torque acting on the rigid
                        body wrt the center of mass.
        """
        F = -body.mass * self.g * self.up
        T = V3.zero()
        return F, T


class Damping(ForceCalculator):
    """
    This class represents a linear damping force type.
    """

    def __init__(self, name):
        """
        Create a damping force instance.

        :param name:   A unique force name among all types of forces.
        """
        super().__init__("Damping", name)
        self.alpha = 0.001  # Linear damping
        self.beta = 0.001  # Angular damping

    def compute(self, body, r, q, v, w):
        """
        Compute the effect of the force type acting on a rigid body.

        :param body:    The reference to the rigid body that the force is acting on. Often used to access
                        information about mass or other properties of the rigid body.
        :param r:       The position of the center of mass of the rigid body.
        :param q:       The orientation of the body frame of the rigid body.
        :param v:       The linear velocity of the center of mass of the rigid body.
        :param w:       The angular velocity of the rigid body.
        :return:        A pair of 3D vectors representing the force and torque acting on the rigid
                        body wrt the center of mass.
        """
        F = -v * self.alpha
        T = -w * self.beta
        return F, T


class Shape:
    """
    A shape class which represents the "geometry" of a rigid body. The shape also has knowledge
    about collision detection data such as signed distance fields (SDFs) which can be shared between
    several rigid bodies and unit-density mass properties of the shape, which can be used for easy
    and quick initialization of the rigid body mass properties.
    """

    def __init__(self, name):
        """
        Create a shape instance with a given name.

        :param name: A unique name that identifies this shape.
        """
        self.name = name
        self.mesh = None  # Polygonal mesh assumed to be in body frame coordinates.
        self.grid = None  # A signed distance field (SDF) in the body frame.
        self.mass = 0.0  # Total mass of shape assuming unit-mass-density.
        self.inertia = (
            V3.zero()
        )  # Body frame inertia tensor assuming unit-mass-density.
        self.r = V3.zero()  # Translation from body frame to model frame.
        self.q = Q.identity()  # Rotation from body frame to model frame.


class RigidBody:
    """
    A rigid body type.
    This class contains all the state and geometric information that is needed by a rigid body simulator.

    As rigid bodies do not change their shape while moving around this is exploited in this simulator to share
    shape (aka geometry or triangle mesh) between many rigid bodies.

    Observe: The k-discrete oriented polytope (k-DOP) bounding volume hierarchies (BVH) that are used for
    collision detection is assuming geometry to be in global world space and not local body space. Hence,
    k-DOP BVHs can not be shared between rigid bodies. This is a limitation specific to our choice of k-DOP
    BVH algorithm. Other approaches for collision detection may be able to re-use and share the collision detection
    data-structures as well as the shape information.
    """

    def __init__(self, name):
        """
        Create a rigid body with a given name.

        :param name:     A unique name that identifies the rigid body instance.
        """
        self.name = name
        self.idx = None  # Unique index of rigid body, used to access body information stored in arrays.
        self.q = Q.identity()  # Orientation stored as a quaternion.
        self.r = V3.zero()  # Center of mass position.
        self.v = V3.zero()  # Linear velocity.
        self.w = V3.zero()  # Angular velocity.
        self.mass = 0.0  # Total mass.
        self.inertia = V3.zero()  # Body frame inertia tensor.
        self.shape = None  # Geometry/Shape of rigid body.
        self.is_fixed = (
            False  # Boolean flag to control if body should be fixed or freely moving.
        )
        self.forces = (
            []
        )  # External forces (like gravity and damping) acting on this body.
        self.material = "default"  # The material this rigid body is made up of.
        self.bvh = None  # k-DOP_bvh encapsulating the entire body.


class ContactPoint:
    """
    A contact point class.

    Contact points represent the geometry between two rigid bodies that come into
    contact. In fact often a set of contact points are used for representing a
    shared interface. One can "think" of the contact points as sample points of
    the contact area between two bodies.
    """

    def __init__(self, bodyA, bodyB, position=V3.zero(), normal=V3.k(), gap=0.0):
        """
        Create an instance of a single contact point.

        Ideally at a point of contact in the real world the gap-value would always be
        perfectly zero. However, in a simulated world we have both approximation, discretization,
        round off and truncation errors which build up during simulations. Hence, often when two
        bodies come into contact they are not exactly touching but can be penetrating or slightly
        separated. The gap-value is used to carry this information. If gap-values are too positive
        then some algorithms may skip the contact information if they consider the value to be too
        big, or other algorithms may add corrective terms to the numerics to fix issues if gap values
        are too negative.

        :param bodyA:    Reference to one of the bodies in contact.
        :param bodyB:    Reference to the other body that is in contact.
        :param position: The 3D "sample" position of the contact point.
        :param normal:   The unit surface normal at the sample position. Assumed to point from A towards B.
        :param gap:      A measure of the penetration/separation between the two bodies.
        """
        if abs(1.0 - V3.norm(normal)) > 0.1:
            raise RuntimeError(
                "ContactPoint.init() was called with non-unit size normal"
            )
        self.bodyA = bodyA
        self.bodyB = bodyB
        self.p = position
        self.n = normal
        self.g = gap


class Parameters:
    """
    This class holds all numerical parameters that controls the simulation behavior.
    """

    def __init__(self):
        """
        Create an instance of the parameter class.
        """
        self.time_step = (
            0.001  # The desired time step size to use when taking one simulation solver step.
        )
        self.max_iterations = 200  # Maximum number of Gauss Seidel iterations
        self.use_bounce = False  # Turning bounce on and off
        self.use_pre_stabilization = (
            False  # Turning pre-stabilization on and off for correcting drift errors
        )
        self.use_post_stabilization = (
            False  # Turning post-stabilization on and off for correcting drift errors
        )
        self.gap_reduction = (
            0.5  # The amount of gap (=penetration) to reduce during stabilization
        )
        self.min_gap_value = (
            0.001  # The minimum allowable gap (=penetration) that will not cause
        )
        self.max_gap_value = (
            0.01  # The maximum possible gap (=penetration) to correct for during
        )
        self.absolute_tolerance = 0.001  # The absolute tolerance value.
        self.relative_tolerance = 0.0001  # The relative tolerance value.
        self.ellipsoid_max_iterations = (
            100  # The maximum number of iterations in the prox ellipsoid binary search
        )
        self.ellipsoid_expansion = 1.5  # The scalar expansion coefficient of the prox ellipsoid binary search interval
        self.ellipsoid_tolerance = (
            10e-10  # The tolerance for the prox ellipsoid binary search
        )
        self.nu_reduce = (
            0.7  # How big a factor to reduce r by if divergence is detected
        )
        self.nu_increase = (
            1.3  # How big a factor to increase r by if convergence is detected
        )
        self.too_small_merit_change = (
            0.01  # The smallest merit change allowed before we increase the r factor
        )
        self.contact_optimization_max_iterations = (
            8  # The maximum number of iterations for optimizing for contacts.
        )
        self.contact_optimization_tolerance = (
            0  # The tolerance for optimizing for contacts.
        )
        self.bvh_chunk_size = 255  # Number of nodes for a k-DOP bvh subtree, a chunk.
        self.K = 3  # The number of directions to use in the k-DOP bounding volumes.
        self.envelope = (
            0.1  # Any geometry within this distance generates a contact point.
        )
        self.resolution = (
            64  # The number of grid cells along each axis in the signed distance fields
        )
        self.proximal_solver = "gauss_seidel" # or "gauss_seidel", "parallel_gauss_seidel", "parallel_jacobi", "parallel_jacboi_hybrid"


class Engine:
    """
    The Engine class holds all data and parameter values that
    describes the current configuration that is being simulated.

    The functions in other modules such as solver, collision detection, api and
    more takes an engine instance as input to provide the function with all the
    information about the world that is being simulated. The information is slightly
    more than just the world state, it is also numerical parameters and force types
    acting in the world etc.
    """

    def __init__(self):
        """
        Create a default empty engine instance.
        """
        self.simulator_type = 'rigid_body' # simulation type for the engine
        self.bodies = dict()
        self.forces = dict()
        self.shapes = dict()
        self.contact_points = []
        self.surfaces_interactions = SurfacesInteractionLibrary()
        self.params = Parameters()
        self.stepper = None  # The time stepper used to simulate the world forward in time.
