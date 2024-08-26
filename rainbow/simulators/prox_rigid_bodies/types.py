"""
This module defines all types used in a rigid body simulation.

The top most type is the Engine type, which is the container of all information used in the simulation of a system of
rigid bodies.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

import numpy as np

import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
from rainbow.geometry.grid3 import Grid
from rainbow.geometry.surface_mesh import Mesh
from rainbow.math.coordsys import CoordSys as JointFrame
from rainbow.geometry.kdop_bvh import Tree


class SurfacesInteraction:
    """
    This class contains parameters that describe the physical
    interaction between two types of material.

    Such as coefficient of friction.
    """

    def __init__(self):
        self.mu: np.ndarray = V3.ones()  # Coefficients of Friction
        self.epsilon: float = 0.0  # Coefficient of restitution


class SurfacesInteractionLibrary:
    """
    This class stores the different combinations of material pair interactions we have created/specified.
    """

    def __init__(self):
        self.storage: dict[tuple[str, str], SurfacesInteraction] = dict()
        self.storage[("default", "default")] = SurfacesInteraction()

    def get_interaction(self, A: str, B: str) -> SurfacesInteraction:
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

    def exist_interaction(self, A: str, B: str) -> bool:
        """
        Test if an interaction instance exists between the two given materials.

        :param A:   Name of first material.
        :param B:   Name of second material.
        :return:    Boolean value indicating if the interaction exists.
        """
        key = (A, B) if A < B else (B, A)
        if key in self.storage:
            return True
        return False

    def exist_material(self, name: str) -> bool:
        """
        Test if a given material exists in the interaction library.
        Meaning that some interaction pair exist with that material name.

        :param name:   The name of the material.
        :return:       Boolean value indicating if the material name was encountered in any interactions.
        """
        for key in self.storage:
            if name in key:
                return True
        return False


class Shape:
    """
    A shape class which represents the "geometry" of a rigid body.

    The shape has knowledge about collision detection data such as signed distance fields (SDFs).

    This data can be shared between several rigid bodies.

    Unit-density mass properties of the shape are used for easy and quick initialization of the rigid body mass
    properties.
    """

    def __init__(self, name: str):
        """
        Create a shape instance with a given name.

        :param name: A unique name that identifies this shape.
        """
        self.name: str = name
        self.mesh: Optional[Mesh] = None  # Polygonal mesh assumed to be in body frame coordinates.
        self.grid: Optional[Grid] = None  # A signed distance field (SDF) in the body frame.
        self.mass: float = 0.0  # Total mass of shape assuming unit-mass-density.
        self.inertia: np.ndarray = (
            V3.zero()
        )  # Body frame inertia tensor assuming unit-mass-density.
        self.r: np.ndarray = V3.zero()  # Translation from body frame to model frame.
        self.q: np.ndarray = Q.identity()  # Rotation from body frame to model frame.


class RigidBody:
    """
    A rigid body type.
    This class contains all the state and geometric information needed by a rigid body simulator.

    As rigid bodies do not change their shape while moving around, this is exploited in this simulator to share
    shape (aka geometry or triangle mesh) between many rigid bodies.

    Observe: The k-discrete oriented polytope (k-DOP) bounding volume hierarchies (BVH) that are used for
    collision detection are assuming geometry to be in global world space and not local body space.

    Hence, k-DOP BVHs cannot be shared between rigid bodies.

    This is a limitation specific to our choice of k-DOP BVH algorithm.

    Other approaches for collision detection may be able to re-use and share the collision detection
    data structures as well as the shape information.
    """

    def __init__(self, name: str):
        """
        Create a rigid body with a given name.

        :param name:     A unique name that identifies the rigid body instance.
        """
        self.name: str = name
        self.idx: Optional[int] = None  # Unique index of rigid body, used to access body information stored in arrays.
        self.q: np.ndarray = Q.identity()  # Orientation stored as a quaternion.
        self.r: np.ndarray = V3.zero()  # Center of mass position.
        self.v: np.ndarray = V3.zero()  # Linear velocity.
        self.w: np.ndarray = V3.zero()  # Angular velocity.
        self.mass: float = 0.0  # Total mass.
        self.inertia: np.ndarray = V3.zero()  # Body frame inertia tensor.
        self.shape: Optional[Shape] = None  # Geometry/Shape of the rigid body.
        self.is_fixed: bool = (
            False  # Boolean flag to control if body should be fixed or freely moving.
        )
        self.forces: list = (
            []
        )  # External forces (like gravity and damping) acting on this body.
        self.material: str = "default"  # The material this rigid body is made up of.
        self.bvh: Optional[Tree] = None  # k-DOP_bvh encapsulating the entire body.


class ForceCalculator(ABC):
    """
    Base class for all force types that can be applied to a rigid body.
    """

    def __init__(self, force_type: str, name: str):
        """
        Create a force instance.

        :param force_type: A unique string telling what kind of force type this is.
        :param name:       A unique name identifying this instance of a "force".
        """
        self.force_type: str = force_type
        self.name: str = name

    @abstractmethod
    def compute(self, body: RigidBody,
                r: np.ndarray, q: np.ndarray, v: np.ndarray, w: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the effect of the force type acting on a rigid body.

        :param body:    The reference to the rigid body that the force is acting on.
                        Often used to access information about mass or other properties of the rigid body.
        :param r:       The position for the center of mass of the rigid body.
        :param q:       The orientation for the body frame of the rigid body.
        :param v:       The linear velocity for the center of mass of the rigid body.
        :param w:       The angular velocity of the rigid body.
        :return:        A pair of 3D vectors representing the force and torque acting on the rigid
                        body wrt the center of mass.
        """
        pass


class Gravity(ForceCalculator):
    """
    This class defines the gravity force type.
    """

    def __init__(self, name: str):
        """
        Create a gravity force instance.

        :param name:   A unique force name among all types of forces.
        """
        super().__init__("Gravity", name)
        self.g: float = 9.81  # Acceleration of gravity
        self.up: np.ndarray = V3.j()  # Up direction

    def compute(self, body: RigidBody,
                r: np.ndarray, q: np.ndarray, v: np.ndarray, w: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the effect of the force type acting on a rigid body.

        :param body:    The reference to the rigid body that the force is acting on.
                        Often used to access information about mass or other properties of the rigid body.
        :param r:       The position for the center of mass of the rigid body.
        :param q:       The orientation for the body frame of the rigid body.
        :param v:       The linear velocity for the center of mass of the rigid body.
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

    def __init__(self, name: str):
        """
        Create a damping force instance.

        :param name:   A unique force name among all types of forces.
        """
        super().__init__("Damping", name)
        self.alpha: float = 0.001  # Linear damping
        self.beta: float = 0.001  # Angular damping

    def compute(self, body: RigidBody,
                r: np.ndarray, q: np.ndarray, v: np.ndarray, w: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the effect of the force type acting on a rigid body.

        :param body:    The reference to the rigid body that the force is acting on.
                        Often used to access information about mass or other properties of the rigid body.
        :param r:       The position for the center of mass of the rigid body.
        :param q:       The orientation for the body frame of the rigid body.
        :param v:       The linear velocity for the center of mass of the rigid body.
        :param w:       The angular velocity of the rigid body.
        :return:        A pair of 3D vectors representing the force and torque acting on the rigid
                        body wrt the center of mass.
        """
        F = -v * self.alpha
        T = -w * self.beta
        return F, T


class ContactPoint:
    """
    A contact point class.

    Contact points represent the geometry between two rigid bodies that come into
    contact.

    In fact, often a set of contact points are used for representing a
    shared interface. One can "think" of the contact points as sample points of
    the contact area between two bodies.
    """

    def __init__(self,
                 bodyA: RigidBody, bodyB: RigidBody,
                 position: np.ndarray = V3.zero(), normal: np.ndarray = V3.k(), gap: float = 0.0
                 ):
        """
        Create an instance of a single contact point.

        Ideally, at a point of contact in the real world, the gap-value would always be
        perfectly zero. However, in a simulated world we have both approximation, discretization,
        round off and truncation errors which build up during simulations. Hence, often when two
        bodies come into contact, they are not exactly touching but can be penetrating or slightly
        separated. The gap-value is used to carry this information. If gap-values are too positive,
        then some algorithms may skip the contact information. Other algorithms may add corrective
        terms to the numerics to fix issues if gap values are too negative.

        :param bodyA:    Reference to one of the bodies in contact.
        :param bodyB:    Reference to the other body that is in contact.
        :param position: The 3D "sample" position of the contact point.
        :param normal:   The surface-normal at the sample position. Assumed to point from A towards B and be a unit
                         vector.
        :param gap:      A measure of the penetration/separation between the two bodies.
        """
        if abs(1.0 - V3.norm(normal)) > 0.1:
            raise RuntimeError(
                "ContactPoint.init() was called with non-unit size normal"
            )
        self.bodyA: RigidBody = bodyA
        self.bodyB: RigidBody = bodyB
        self.p: np.ndarray = position
        self.n: np.ndarray = normal
        self.g: float = gap


class Hinge:
    """
    A hinge joint class.

    A hinge joint is a revolute joint between two rigid bodies.
    """

    def __init__(self, name: str):
        """
        Create an instance of a single hinge joint.

        :param name:    The name of the new hinge joint.
        """
        self.name: str = name
        self.idx: Optional[
            int] = None  # Unique index of hinge joint, used to access hinge information stored in arrays.
        self.parent: Optional[RigidBody] = None  # Reference to the parent link of the hinge.
        self.child: Optional[RigidBody] = None  # Reference to the child link of the hinge
        self.socket_p: Optional[JointFrame] = None  # Joint frame on body A wrt body A's local body frame
        self.socket_c: Optional[JointFrame] = None  # Joint frame on body B wrt body B's local body frame
        self.arm_p: Optional[np.ndarray] = None
        self.arm_c: Optional[np.ndarray] = None
        self.axis_p: Optional[np.ndarray] = None  # Hinge axis wrt local coordinate frame of the parent link.
        self.axis_c: Optional[np.ndarray] = None  # Hinge axis wrt local coordinate frame of the child link.

    def set_parent_socket(self, body: RigidBody, socket: JointFrame) -> None:
        """
        Set the parent link information of the joint.

        :param body:        A reference to the rigid body that will be the parent link of the joint.
        :param socket:      A reference to the joint frame that will define the joint socket on the parent link.
        :return:            None.
        """
        self.parent = body
        self.socket_p = socket
        self.arm_p = socket.r.copy()
        self.axis_p = Q.rotate(socket.q, V3.k())

    def set_child_socket(self, body: RigidBody, socket: JointFrame) -> None:
        """
        Set the child link information of the joint.

        :param body:        A reference to the rigid body that will be the child link of the joint.
        :param socket:      A reference to the joint frame that will define the joint socket on the child link.
        :return:            None.
        """
        self.child = body
        self.socket_c = socket
        self.arm_c = socket.r.copy()
        self.axis_c = Q.rotate(socket.q, V3.k())


class Parameters:
    """
    This class holds all numerical parameters that control the simulation behavior.
    """

    def __init__(self):
        """
        Create an instance of the parameter class.
        """
        self.time_step: float = (
            0.01  # The desired time step size to use when taking one simulation solver step.
        )
        self.max_iterations: int = 200  # Maximum number of Gauss Seidel iterations
        self.use_bounce: bool = False  # Turning bounce on and off
        self.use_pre_stabilization: bool = (
            False  # Turning pre-stabilization on and off for correcting drift errors
        )
        self.use_post_stabilization: bool = (
            False  # Turning post-stabilization on and off for correcting drift errors
        )
        self.gap_reduction: float = (
            0.5  # The amount of gap (=penetration) to reduce during stabilization
        )
        self.min_gap_value: float = (
            0.001  # The minimum allowable gap (=penetration) that will not cause
        )
        self.max_gap_value: float = (
            0.01  # The maximum possible gap (=penetration) to correct for during
        )
        self.absolute_tolerance: float = 0.001  # The absolute tolerance value.
        self.relative_tolerance: float = 0.0001  # The relative tolerance value.
        self.ellipsoid_max_iterations: int = (
            100  # The maximum number of iterations in the prox ellipsoid binary search
        )
        self.ellipsoid_expansion: float = (
            1.5  # The scalar expansion coefficient of the prox ellipsoid binary search interval
        )
        self.ellipsoid_tolerance: float = (
            10e-10  # The tolerance for the prox ellipsoid binary search
        )
        self.nu_reduce: float = (
            0.7  # How big a factor to reduce r by if divergence is detected
        )
        self.nu_increase: float = (
            1.3  # How big a factor to increase r by if convergence is detected
        )
        self.too_small_merit_change: float = (
            0.01  # The smallest merit change allowed before we increase the r factor
        )
        self.contact_optimization_max_iterations: int = (
            8  # The maximum number of iterations for optimizing for contacts.
        )
        self.contact_optimization_tolerance: float = (
            0  # The tolerance for optimizing for contacts.
        )
        self.bvh_chunk_size: int = 255  # Number of nodes for a k-DOP bvh subtree, a chunk.
        self.K: int = 3  # The number of directions to use in the k-DOP bounding volumes.
        self.envelope: float = (
            0.1  # Any geometry within this distance generates a contact point.
        )
        self.resolution: int = (
            64  # The number of grid cells along each axis in the signed distance fields
        )


class Engine:
    """
    The Engine class holds all data and parameter values that
    describe the current configuration that is being simulated.

    The functions in other modules such as solver, collision detection, api and
    more take an engine instance as input to provide the function with all the
    information about the world that is being simulated. The information is slightly
    more than just the world state, it is also numerical parameters and force types
    acting in the world, etc.
    """

    def __init__(self):
        """
        Create a default empty engine instance.
        """
        self.simulator_type: str = 'rigid_body'  # The simulation type for the engine
        self.bodies: dict[str, RigidBody] = dict()
        self.forces: dict[str,  Union[Gravity, Damping]] = dict()
        self.shapes: dict[str, Shape] = dict()
        self.hinges: dict[str, Hinge] = dict()  # All hinge joints in the current simulation
        self.contact_points: list[ContactPoint] = []
        self.surfaces_interactions = SurfacesInteractionLibrary()
        self.params = Parameters()
        self.stepper = None  # The time stepper used to simulate the world forward in time.
