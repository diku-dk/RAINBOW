
class GearFactory:

    @staticmethod
    def _roll_angle(r_base: float, r_top: float):
        """
        Compute the roll angle.

        :param r_base:  The radius of the inner circle.
        :param r_top:   The radius of the outer circle.
        :return:        The roll angle of the inner circle that will create an involute curve connecting to the outer circle.
        """
        return np.sqrt((r_top / r_base) ** 2 - 1)

    @staticmethod
    def _span_angle(roll_angle: float):
        """
        This function computes the span angle of the involute curve. That is if
        projected onto its base circle how big an angle does the resulting
        circle arch span?

        :param roll_angle:
        :return:
        """
        x = (np.cos(roll_angle) + roll_angle * np.sin(roll_angle))
        y = (np.sin(roll_angle) - roll_angle * np.cos(roll_angle))
        return np.arctan2(y, x)

    @staticmethod
    def _make_involute_curve(r_base: float, r_top: float, shift: float = 0, reverse: bool = False):
        """
        This function is used to create involute curve.
        Let us look at the involute curve going in the ccw direction (we call this the forward direction).
        The opposing direction we refer to the reverse direction.

        The direction that the involute curve should wound around the inner circle. CCW is forward and CW is reserve.


        A point on the involute curve is given by

            q(theta) = r_base (cos(theta), sin(theta))^T + r_base theta (sin(theta), -cos(theta))^T

        Solving for the root of

            norm(q(beta))^2 - r_top^2 = 0

        Gives us the roll angle beta. This function create the involute curve going from 0 to beta which will
        connect the circle with radius r_base to the circle with radius r_top.

        :param r_base:  The radius of the inner circle.
        :param r_top:   The radius of the outer circle.
        :param shift:   The shifting angle for where the involute curve should start on the inner circle.
        :param reverse: A boolean flag indicating if the involute curve should be reversed.
        :return:        A curve represented as tuple of two numpy arrays. The First one is x-coordinates, and the second is y-coordinates of points on the curve.
        """
        theta = np.linspace(0, GearFactory._roll_angle(r_base, r_top), 12)
        if reverse:
            theta = -theta[::-1]
        ix = r_base * (np.cos(theta + shift) + theta * np.sin(theta + shift))
        iy = r_base * (np.sin(theta + shift) - theta * np.cos(theta + shift))
        return ix, iy

    @staticmethod
    def make_gear_specs(m: float = 1.0, Z: int = 12, alpha: float = 20.0):
        """
        This function is used to create a gear geometry profile.

        For assembling gears remember that mating gears must share the same module. They
        must also have the same pressure angle. Their pitch circles must be tangent.

        The code is based on descriptions from https://www.stlgears.com/ and https://khkgears.net/new/gear_knowledge/

        :param m: Module value
        :param Z:  Number of teeth of the gear
        :param alpha: The pressure angle given in units of degrees

        :return: A tuple with the first element being the pitch circle radius.
        """
        # First, we compute a lot of gear parameters that are used for defining the geometry of the gear.
        rad = np.pi * alpha / 180.0  # Convert pressure angle to radians
        h_a = m  # The distance from the reference circle to tooth tip
        h_d = 1.25 * m  # Distance from reference circle to root of the teeth
        h = 2.25 * m  # Total tooth height
        p = np.pi * m  # The distance between corresponding points on reference circle for adjacent tooths
        s = p / 2  # Tooth thickness on reference circle
        R_p = m * Z / 2.0  # Radius of the reference circle also known as the pitch circle
        R_r = R_p - h_d  # Radius of the dedendum circle (root circle)
        R_t = R_p + h_a  # Radius of the addendum circle (tip circle)
        R_b = R_p * np.cos(rad)  # Radius of the base circle which is the basis for the involute curve

        # Generate teeth profile
        pitch_roll = GearFactory._roll_angle(R_b, R_p)
        top_roll = GearFactory._roll_angle(R_b, R_t)

        delta = GearFactory._span_angle(
            pitch_roll)  # Angle spanned by involute curve going from base circle to pitch circle
        beta = np.pi / Z  # Angle spanned by tooth thickness at the pitch circle
        gamma = GearFactory._span_angle(
            top_roll) - delta  # Angle spanned by involute curve going from pitch circle to top circle

        corners = []
        gear_x_list = []
        gear_y_list = []
        for z in range(Z):
            shift = z * beta * 2
            forward_shift = shift - delta
            reverse_shift = shift + delta + beta
            # Generate the involute curve going from base circle to the top circle
            forward_x, forward_y = GearFactory._make_involute_curve(r_base=R_b, r_top=R_t, shift=forward_shift)
            # Generate top circle arch between the forward and reverse involute curves
            start_top_theta = shift + gamma
            end_top_theta = shift + beta - gamma
            theta = np.linspace(start_top_theta, end_top_theta, 6)
            top_x = R_t * np.cos(theta)
            top_y = R_t * np.sin(theta)
            # Generate the involute curve going from top circle to the base circle
            rev_x, rev_y = GearFactory._make_involute_curve(r_base=R_b, r_top=R_t, shift=reverse_shift, reverse=True)
            # Generate the circular arch on root circle from this tooth end to the next tooth start point.
            start_root_theta = shift + beta + delta
            end_root_theta = shift + 2 * beta - delta
            theta = np.linspace(start_root_theta, end_root_theta, 6)
            root_x = R_r * np.cos(theta)
            root_y = R_r * np.sin(theta)
            # Piece all the tooth curves together to one curve
            tooth_x = np.concatenate((forward_x, top_x[1:-1], rev_x, root_x))
            tooth_y = np.concatenate((forward_y, top_y[1:-1], rev_y, root_y))
            # Put indices of tooth root vertices into a list for later
            tooth_start_idx = len(gear_x_list) - 1
            tooth_end_idx = len(gear_x_list) + len(tooth_x) - len(root_x)
            corners.append([tooth_start_idx, tooth_end_idx])
            # Append the tooth curve to all previous generated teeth
            gear_x_list.extend(tooth_x)
            gear_y_list.extend(tooth_y)

        gear_x = np.array(gear_x_list, dtype=float)
        gear_y = np.array(gear_y_list, dtype=float)

        # Store all the technical specifications about the gear
        specs = {}
        specs['pitch radius'] = R_p
        specs['top radius'] = R_t
        specs['base radius'] = R_b
        specs['root radius'] = R_r
        specs['tooth thickness'] = s
        specs['pitch'] = p
        specs['tooth height'] = h
        specs['module'] = m
        specs['number of teeth'] = Z
        specs['pressure angle'] = alpha
        specs['beta'] = beta  # Tooth thickness angle span on pitch circle
        specs['gamma'] = gamma  # Involute angle span from pitch to top circle
        specs['delta'] = delta  # Involute angle span from base to pitch circle
        specs['x'] = gear_x  # The x coordinates of the gear profile
        specs['y'] = gear_y  # The y coordinates of the gear profile
        specs['corners'] = np.array(corners, dtype=int)
        return specs

    @staticmethod
    def make_gears_assembly(drive_gear: dict
                            , driven_gear: dict
                            , cx: float
                            , cy: float
                            , theta: float
                            , omega: float
                            ):
        """

        :param drive_gear:   A dictionary with specifications of the driving gear.
        :param driven_gear:  A dictionary with specifications of the driven gear.
        :param cx:           The world x coordinate of the center position of the drive gear.
        :param cy:           The world y coordinate of the center position of the drive gear.
        :param theta:        Current rotation of the drive gear
        :param omega:        The connection angle to the drive gear
        :return:
        """
        PR1 = drive_gear['pitch radius']
        PR2 = driven_gear['pitch radius']
        M1 = drive_gear['module']
        M2 = driven_gear['module']
        PA1 = drive_gear['pressure angle']
        PA2 = driven_gear['pressure angle']
        Z1 = drive_gear['number of teeth']
        Z2 = driven_gear['number of teeth']
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
        # It must hold that if contact is not slipping between the two gears
        # then they "travel" the same distance as they rotate. So we must have
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
        # If both gears are in neutral positions then rotating gear 2 by pi will align teeth of both gears.
        # However, gear 1 is not in neutral position, and gear 2 is not connecting to gear 1 with an angle of zero.
        # Hence, we must compute the relative connection angle of gear 2 to the current world location
        # of gear1's neutral position.
        phi = np.pi - ratio * (theta - omega) + omega
        return tx, ty, phi

    @staticmethod
    def make_gear_mesh(gear: dict, face_width: float = 1.0):
        x = gear['x']
        y = gear['y']
        z = np.ones_like(x) * face_width / 2

        V = np.row_stack(
            (
                np.column_stack((x, y, -z))
                , np.column_stack((x, y, z))
            )
        )

        N = len(x)
        T = []
        # Create mesh of teeth profile of the gear
        for i in range(N):
            j = (i + 1) % N
            k = N + j
            m = N + i
            T.append([i, j, k])
            T.append([i, k, m])

        # Create top and bottom caps of gear
        C = gear['corners']
        for start, stop in C:
            # First, we do bottom teeth
            indices = np.arange(start, stop + 1, 1, dtype=int)
            indices[indices < 0] += N

            tooth_center = np.mean(V[indices, :], axis=0)
            V = np.row_stack((V, tooth_center))
            idx_tooth_center = V.shape[0] - 1
            M = len(indices)
            for m in range(M):
                j = indices[m]
                k = indices[(m + 1) % M]
                T.append([idx_tooth_center, k, j])

            # Second, we do top teeth
            indices += N
            tooth_center = np.mean(V[indices, :], axis=0)
            V = np.row_stack((V, tooth_center))
            idx_tooth_center = V.shape[0] - 1
            M = len(indices)
            for m in range(M):
                j = indices[m]
                k = indices[(m + 1) % M]
                T.append([idx_tooth_center, j, k])

        cx = np.mean(x)
        cy = np.mean(y)
        bottom_center = np.array([cx, cy, -face_width / 2]).T
        top_center = np.array([cx, cy, face_width / 2]).T
        V = np.row_stack((V, bottom_center, top_center))
        idx_top = V.shape[0] - 1
        idx_bot = V.shape[0] - 2
        for start, stop in C:
            if start < 0:
                start += N
            T.append([idx_bot, stop, start])
            T.append([idx_top, start + N, stop + N])

        K = len(C)
        for k in range(K):
            start = C[k, 1]
            stop = C[(k + 1) % K, 0]
            if start < 0:
                start += N
            if stop < 0:
                stop += N
            for i in range(start, stop):
                j = i + 1
                T.append([idx_bot, j, i])
            for i in range(start, stop):
                j = i + 1
                T.append([idx_top, i + N, j + N])
        T = np.array(T, dtype=int)
        return V, T


def create_gear_train(engine
                      , N: int
                      , density: float = 1.0
                      , material_name: str = "default"
                      ):
    body_names = []
    gear_specs = []
    gear_names = []

    q_m2w = Q.Rx(-np.pi / 2)  # Needed to change z-up direction to y-up direction

    m = 1.0  # Gear module
    numbers = [7, 12, 15, 28]  # Possible gear teeth to use
    Z = np.random.choice(numbers, size=(N,))  # Random list of N numbers of gear teeth
    alpha = 20  # Pressure angle
    face_width = 10.0  # Width of the gear

    for i in range(N):
        specs = GearFactory.make_gear_specs(m=m, Z=Z[i], alpha=alpha)
        shape_name = API.generate_unique_name("shape")
        body_name = API.generate_unique_name("body")

        body_names.append(body_name)
        gear_names.append(shape_name)
        gear_specs.append(specs)

        V, T = GearFactory.make_gear_mesh(specs, face_width=face_width)

        mesh = API.create_mesh(V, T)
        API.create_shape(engine, shape_name, mesh)

        API.create_rigid_body(engine, body_name)
        API.connect_shape(engine, body_name, shape_name)

        r_m = V3.make(0.0, 0.0, 0.0)  # Model space position of gear (z-up)
        q_m = Q.identity()  # Model space orientation of gear (z-up)
        r_w = Q.rotate(q_m2w, r_m)  # World position of gear (y-up)
        q_w = Q.prod(q_m2w, q_m)  # World orientation of gear (y-up)

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
        r_m = V3.make(cx, cy, cz)  # Model space position of gear (z-up)
        q_m = Q.Rz(theta)  # Model space orientation of gear (z-up)
        r_w = Q.rotate(q_m2w, r_m)  # World position of gear (y-up)
        q_w = Q.prod(q_m2w, q_m)  # World orientation of gear (y-up)
        driven_gear_body_name = body_names[i + 1]
        API.set_position(engine, driven_gear_body_name, r_w, True)
        API.set_orientation(engine, driven_gear_body_name, q_w, True)

    return body_names

