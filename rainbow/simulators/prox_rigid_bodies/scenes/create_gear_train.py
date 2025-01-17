"""
This script contains code to create gear shapes and gear mechanisms.
"""

import numpy as np

import rainbow.math.involute as INV
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.simulators.prox_rigid_bodies.api as API
import rainbow.geometry.surface_mesh as MESH
from rainbow.simulators.prox_rigid_bodies.types import Engine

#from rainbow.procedural.gears.types import Gear, GearSpec
import rainbow.procedural.gears.generators as GEAR


class GearFactory:
    """
    This class contains methods to create gear shapes and gear mesh.
    """

    @staticmethod
    def make_gears_assembly(drive_gear: GEAR.GearSpec,
                            driven_gear: GEAR.GearSpec,
                            cx: float,
                            cy: float,
                            theta: float,
                            omega: float
                            ) -> tuple[float, float, float]:
        """
        This function pairs a drive gear with a driven gear.

        It assumes one knows the position and orientation of the drive gear and then the function computes
        the position and orientation of the driven gear given a "heading" or "connection" angle between
        the two gear centers.

        :param drive_gear:   A dictionary with specifications of the driving gear.
        :param driven_gear:  A dictionary with specifications of the driven gear.
        :param cx:           The world x coordinate for the center position of the drive gear.
        :param cy:           The world y coordinate for the center position of the drive gear.
        :param theta:        Current rotation of the drive gear.
        :param omega:        The connection angle to the drive gear.
        :return:             The position and orientation of the driven gear.
        """
        PR1 = drive_gear.rp # pitch radius
        PR2 = driven_gear.rp # pitch radius
        M1 = drive_gear.m # module
        M2 = driven_gear.m # module
        PA1 = drive_gear.alpha # pressure angle
        PA2 = driven_gear.alpha # pressure angle
        Z1 = drive_gear.z # number of teeth
        Z2 = driven_gear.z # number of teeth
        if M1 != M2:
            raise ValueError("Gears are not compatible, must have same module")
        if PA1 != PA2:
            raise ValueError("Gears are not compatible, must have same pressure angles")

        R = PR1 + PR2
        tx = np.cos(omega) * R + cx
        ty = np.sin(omega) * R + cy
        # If gear 1 rotates an angle beta1, then gear 2 will rotate beta2 = - ratio * beta1
        # where ration= Z1/Z2
        #
        # Pitch circle 1 has radius
        #
        #   r1 =  m * Z1 / 2.0
        #
        # Pitch circle 2 has radius
        #
        #   r2 =  m * Z2 / 2.0
        #
        # If contact is not slipping between the two gears, then it must hold that
        # they "travel" the same distance as they rotate.
        #
        # So we must have
        #
        #   beta2 r2 = - beta1 r1
        #
        # The minus sign is there because the two gears are always rotating
        # opposite. So if beta1>0 then beta2<0 and vice versa.
        # Now substitution gives us
        #
        #   beta2 Z2 = - beta1 Z1
        #
        # And we find
        #
        #   beta2 = - (Z1/Z2) beta1
        #
        ratio = Z1 / Z2
        # If both gears are in neutral positions, then rotating gear 2 by pi will align teeth of both gears.
        # However, gear 1 is not in neutral position, and gear 2 is not connected to gear 1 with an angle of zero.
        # Hence, we must compute the relative connection angle of gear 2 to the current world location
        # of gear1's neutral position.
        phi = np.pi - ratio * (theta - omega) + omega
        return tx, ty, phi


def create_gear_train(engine: Engine,
                      N: int,
                      density: float = 1.0,
                      material_name: str = "default"
                      ) -> list[str]:
    """
    This function creates a sequence of N gears.

    All gears are paired up to form a chain of gears.

    :param engine:                    The engine that will be used to create the dry stone rigid bodies in.
    :param N:                         The number of gears to create in the scene.
    :param density:                   The mass density to use for all the rigid bodies.
    :param material_name:             The material name to use for all the rigid bodies that are created.
    :return:                          A list with the names of all the rigid bodies that were created.
    """
    body_names = []
    gear_specs = []
    gear_names = []

    # Create and place all the gears
    q_m2w = Q.Rx(-np.pi / 2)  # Needed to change the z-up direction to a y-up direction.

    m = 1.0  # Gear module
    numbers = np.array([7, 12, 15, 28, 35, 41, 55, 65, 74, 89, 101], dtype=int)  # Possible gear teeth to use.
    Z = np.random.choice(numbers, size=(N,))  # An N-long random list of gear teeth values.
    alpha = 20  # Pressure angle.
    face_width = 10.0  # Width of the gear.
    
    gear_factory = GEAR.GearFactory()

    for i in range(N):
        spec = GEAR.GearSpec(m, Z[i], alpha, helix_angle=20)

        shape_name = API.generate_unique_name("shape")
        body_name = API.generate_unique_name("body")

        body_names.append(body_name)
        gear_names.append(shape_name)
        gear_specs.append(spec)
        
        gear = gear_factory.create_gear(spec, face_width)

        mesh = API.create_mesh(gear.V, gear.T)
        API.create_shape(engine, shape_name, mesh)

        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_m = V3.make(0.0, 0.0, 0.0)  # Model space position of gear (z-up).
        q_m = Q.identity()  # Model space orientation of gear (z-up).
        r_w = Q.rotate(q_m2w, r_m)  # World position of gear (y-up).
        q_w = Q.prod(q_m2w, q_m)  # World orientation of gear (y-up).

        API.set_position(engine, body_name, r_w, True)
        API.set_orientation(engine, body_name, q_w, True)
        API.set_body_type(engine, body_name, "free")
        API.set_body_material(engine, body_name, material_name)
        API.set_mass_properties(engine, body_name, density)

    cx = 0
    cy = 0
    cz = 0
    theta = 0
    for i in range(N - 1):
        drive_gear = gear_specs[i]
        driven_gear = gear_specs[i + 1]
        omega = np.random.uniform(0, 2 * np.pi)
        cx, cy, theta = GearFactory.make_gears_assembly(drive_gear, driven_gear, cx, cy, theta, omega)
        cz += face_width / 2
        r_m = V3.make(cx, cy, cz)  # Model space position of gear (z-up).
        q_m = Q.Rz(theta)  # Model space orientation of gear (z-up).
        r_w = Q.rotate(q_m2w, r_m)  # World position of gear (y-up).
        q_w = Q.prod(q_m2w, q_m)  # World orientation of gear (y-up).
        driven_gear_body_name = body_names[i + 1]
        API.set_position(engine, driven_gear_body_name, r_w, True)
        API.set_orientation(engine, driven_gear_body_name, q_w, True)

    # Create a fixed object in the world
    shape_name = API.generate_unique_name("ground_shape")

    V, T = MESH.create_box(200.0, 1.0, 200.0)
    mesh = API.create_mesh(V, T)
    API.create_shape(engine, shape_name, mesh)

    body_name = API.generate_unique_name("ground_body")
    API.create_rigid_body(engine, body_name)
    API.connect_shape(engine, body_name, shape_name)

    r = V3.make(0.0, -0.5 - face_width, 0.0)
    q = Q.identity()

    API.set_position(engine, body_name, r, True)
    API.set_orientation(engine, body_name, q, True)

    API.set_body_type(engine, body_name, "fixed")
    API.set_body_material(engine, body_name, material_name)
    API.set_mass_properties(engine, body_name, density)
    body_names.append(body_name)

    # Create hinge-joints between gears and fixed object
    parent_name = body_name
    for i in range(N):
        child_name = body_names[i]
        hinge_name = parent_name + "_" + child_name
        API.create_hinge(engine, hinge_name)
        origin = API.get_position(engine, child_name) - V3.make(0.0, face_width/2.0, 0.0)
        API.set_hinge(
            engine=engine,
            hinge_name=hinge_name,
            parent_name=parent_name,
            child_name=child_name,
            origin= origin,
            axis= V3.k(),
            mode="world"
        )

    return body_names
