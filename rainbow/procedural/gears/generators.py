import numpy as np

import rainbow.math.involute as INVOLUTE
import rainbow.procedural.gears.mating as MATING
import rainbow.geometry.surface_mesh as MESH

from .types import Gear, GearSpec, PlanetaryGearSpec, PlanetaryGear


class GearFactory:
    def __init__(self, side_points: int = 10, top_points: int = 1, bottom_points: int = 1) -> None:
        """Creates a new gear factory.
        
        :param side_points: The number of side points for the gear.
        :param top_points: The number of top points for the gear.
        :param bottom_points: The number of bottom points for the gear.
        """
        
        self.side_points = side_points
        self.top_points = top_points
        self.bottom_points = bottom_points
    
    def create_gear(self, spec: GearSpec, face_width: float, subdivisions: int = 3) -> Gear:
        """Creates the mesh for an involute gear.
        
        :param spec: The gear specification.
        :param face_width: The face width of the gear.
        :param subdivisions: The number of subdivisions for the face width.
        :return: The gear vertices and triangles.
        """
        
        V_profile = self._create_profile_points(spec)
        V_cylinder = self._create_cylinder_points(spec)
        
        V, T = self._generate_mesh(spec, V_profile, V_cylinder)
        V, T = self._extrude(spec, V, T, face_width, subdivisions)
        
        if spec.is_bevel:
            V, T = self._bevel_transformation(spec, V, T, face_width)
        
        return Gear(spec, MESH.Mesh(V, T))
    
    def create_planetary_gear(self, planetary_spec: PlanetaryGearSpec, face_width: float, subdivisions: int = 3) -> PlanetaryGear:
        # Create the gears
        sun_gear = self.create_gear(planetary_spec.sun_spec, face_width, subdivisions)
        planet_gears = [self.create_gear(planetary_spec.planet_spec, face_width, subdivisions) for _ in range(planetary_spec.N_planet)]
        ring_gear = self.create_gear(planetary_spec.ring_spec, face_width, subdivisions)
        
        # Update the position and orientation of the planet gears so they mesh with the sun gear
        for planet_gear, angle in zip(planet_gears, planetary_spec.planet_angles):
            planet_gear.position = MATING.compute_gear_position(sun_gear.spec, planet_gear.spec, angle)
            planet_gear.orientation = MATING.compute_gear_orientation(sun_gear.spec, planet_gear.spec, angle)
        
        # Update the orientation of the ring gear so it meshes with the planet gears
        ring_gear.orientation = MATING.compute_internal_gear_orientation(ring_gear.spec)
        
        return PlanetaryGear(planetary_spec, sun_gear, planet_gears, ring_gear)

    def points_per_tooth(self, spec: GearSpec) -> int:
        """Calculates the number of points per tooth for the gear.
        
        :param spec: The gear specification.
        :return: The number of points per tooth for the gear.
        """
        
        points = 2 * self.side_points # side points, left and right
        points += self.top_points # top points
        
        # bottom points
        if spec.rb >= spec.rd:
            # if base circle is greater than or equal to dedendum circle
            points += self.bottom_points + 2
        else:
            # if base circle is less than dedendum circle
            points += self.bottom_points
        
        return points

    def total_gear_points(self, spec: GearSpec) -> int:
        """Calculates the total number of points for the gear.
        
        :param spec: The gear specification.
        :return: The total number of points for the gear.
        """
        
        return spec.z * self.points_per_tooth(spec)

    def _create_profile_points(self, spec: GearSpec) -> np.ndarray:
        """Creates the gear profile vertices for the gear.
        
        :param spec: The gear specification.
        :return: The gear profile vertices.
        """
        
        ts = np.linspace(spec.t_min, spec.t_max, self.side_points)
        rev_ts = -ts[::-1]
        
        V = []
        for idx in range(spec.z):
            theta_i = spec.theta_i[idx]
            theta_i_next = spec.theta_i[(idx + 1) % spec.z]
            if theta_i_next < theta_i:
                theta_i_next += 2 * np.pi
            
            right_root_offset = theta_i - spec.gamma_b / 2
            left_root_offset = theta_i + spec.gamma_b / 2
            
            # Right side of the tooth
            right_inv = INVOLUTE.InvoluteCurve(spec.rb, right_root_offset)
            V.append(right_inv(ts))
            
            # Top
            right_top_offset = theta_i - spec.gamma_a / 2
            left_top_offset = theta_i + spec.gamma_a / 2
            top_theta = np.linspace(right_top_offset, left_top_offset, self.top_points + 2)[1:-1]
            V.append(np.vstack((spec.ra * np.cos(top_theta), spec.ra * np.sin(top_theta))).T)
            
            # Left side of the tooth
            left_inv = INVOLUTE.InvoluteCurve(spec.rb, left_root_offset)
            V.append(left_inv(rev_ts))
            
            # Bottom
            if spec.rb >= spec.rd:
                bottom_theta = np.linspace(left_root_offset, left_root_offset + 2 * spec.gamma_p - spec.gamma_b, self.bottom_points + 2)
                V.append(np.vstack((spec.rd * np.cos(bottom_theta), spec.rd * np.sin(bottom_theta))).T)
            else:
                bottom_start = theta_i + spec.gamma_d / 2
                bottom_end = theta_i_next - spec.gamma_d / 2
                bottom_theta = np.linspace(bottom_start, bottom_end, self.bottom_points + 2)[1:-1]
                V.append(np.vstack((spec.rd * np.cos(bottom_theta), spec.rd * np.sin(bottom_theta))).T)
        
        # Add third dimension to the vertices
        V = np.vstack(V)
        V = np.hstack((V, np.zeros((len(V), 1))))
        
        return V

    def _create_cylinder_points(self, spec: GearSpec) -> np.ndarray:
        """Creates the cylinder vertices for the gear.
        
        :param spec: The gear specification.
        :return: The cylinder vertices.
        """
        
        # Compute the radius of the cylinder
        r = self._get_cylinder_radius(spec)
        
        # Compute the angles for the cylinder vertices
        theta = np.linspace(0, 2 * np.pi, spec.z + 1)[:-1]
        if spec.is_internal:
            theta += np.pi / spec.z
        
        # Create the cylinder vertices
        V = np.vstack((r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta))).T
        
        return V
    
    def _get_cylinder_radius(self, spec: GearSpec) -> float:
        """Computes the radius of the cylinder for the gear.
        
        :param spec: The gear specification.
        :return: The radius of the cylinder.
        """
        
        if spec.is_internal:
            return 1.2 * spec.ra
        elif spec.is_bevel:
            return 0.9 * spec.rd
        else:
            return 0.2 * spec.rd

    def _generate_mesh(self, spec: GearSpec, V_profile: np.ndarray, V_cylinder: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tessellates the gear profile and cylinder to create the gear teeth.
        
        :param spec: The gear specification.
        :param V_profile: The gear profile vertices.
        :param V_cylinder: The gear cylinder vertices.
        :return: The gear vertices and triangles.
        """
        
        total_gear_points = self.total_gear_points(spec)
        points_per_tooth = self.points_per_tooth(spec)
        
        V = np.vstack((V_profile, V_cylinder))
        T = []

        # Compute the indices for the profile and cylinder vertices
        # The index shift is used to align the profile vertices with the cylinder vertices
        index_shift = (points_per_tooth - self.top_points) // 2 if spec.is_internal else 0
        # Compute the indices for the profile and cylinder vertices
        profile_indices = np.array([[i + j * points_per_tooth for i in range(points_per_tooth)] for j in range(spec.z)])
        profile_indices = (profile_indices + index_shift) % total_gear_points
        circle_indices = np.array(range(total_gear_points, total_gear_points + spec.z))
        
        # Create the triangles for the gear teeth
        for k, (ci, pi) in enumerate(zip(circle_indices, profile_indices)):
            T.append((ci, pi[0], profile_indices[k-1][-1]))
            T.append((ci, profile_indices[k-1][-1], circle_indices[k-1]))
            for i in range(points_per_tooth - 1):
                j1 = (i + 1) % points_per_tooth
                T.append((ci, pi[j1], pi[i]))
        
        return V, np.array(T)

    def _extrude(self, spec: GearSpec, V: np.ndarray, T: np.ndarray, face_width: float, subdivisions: int) -> np.ndarray:
        """Extrudes the gear profile to create the gear teeth.
        
        :param spec: The gear specification.
        :param V: The gear profile vertices.
        :param T: The gear profile triangles.
        :param face_width: The face width of the gear.
        :param subdivisions: The number of subdivisions for the face width.
        :return: The gear vertices and triangles.
        """
        
        face_width_step = face_width / (subdivisions + 1)
        
        # Create the vertices and triangles for each subdivision
        Vs = [np.copy(V)]
        Ts = [np.copy(T)]
        for i in range(subdivisions + 1):
            # Transform the profile. This translates and rotates the profile
            Vi = self._transform_profile(spec, V, face_width_step, i, subdivisions)
            # Connect the transformed profile to the previous profile
            Ti = self._connect_profiles(spec, len(V), len(V) * i)
            Vs.append(Vi)
            Ts.append(Ti)
        
        T2 = T[:, ::-1] + len(V) * (subdivisions + 1)
        Ts.append(T2)
        
        V_final = np.vstack(Vs)
        V_final[:, 2] -= face_width / 2
        
        T_final = np.vstack(Ts)
        if spec.is_internal:
            T_final = T_final[:,::-1]
        
        return V_final, T_final

    def _connect_profiles(self, spec: GearSpec, total_points: int, index_shift: int):
        """Connects the gear profile vertices to form the gear teeth.
        
        :param spec: The gear specification.
        :param total_points: The total number of points per tooth.
        :param index_shift: The index shift for the profile vertices.
        :return: The triangles connecting the gear profile vertices.
        """
        
        total_gear_points = self.total_gear_points(spec)
        
        T_profile = []
        for idx in range(total_gear_points):
            i = idx + index_shift
            j1 = i + total_points
            j2 = (idx + 1) % total_gear_points + index_shift
            k = ((idx + 1) % total_gear_points) + total_points + index_shift
            T_profile.append((i, k, j1))
            T_profile.append((i, j2, k))

        T_cylinder = []
        for idx in range(spec.z):
            idx = idx + total_gear_points + index_shift
            j1 = idx + total_points
            j2 = (idx + 1) % spec.z + total_gear_points + index_shift
            k = ((idx + 1) % spec.z) + total_gear_points + total_points + index_shift
            T_cylinder.append((idx, j1, k))
            T_cylinder.append((idx, k, j2))
        
        return np.vstack((T_profile, T_cylinder))

    def _transform_profile(self, spec: GearSpec, V: np.ndarray, face_width_step: float, i: int, subdivisions: int) -> np.ndarray:
        """Transforms the gear profile vertices by updating the z coordinate and optionally rotating the profile.
        
        :param spec: The gear specification.
        :param V: The gear profile vertices.
        :param T: The gear profile triangles.
        :param face_width_step: The step size for the face width.
        :param i: The current subdivision index.
        :param subdivisions: The total number of subdivisions.
        :return: The transformed gear profile vertices.
        """
        
        # Create the hommogenous coordinates
        hommogenous = np.hstack((V, np.ones((len(V), 1))))
        
        # Create the rotation matrix
        z = (i + 1) * face_width_step
        theta = 0
        if spec.beta is not None:
            vertical_pitch = 2 * np.pi * spec.rp * np.tan(spec.beta)
            theta = z / vertical_pitch
        
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])
        
        # Transform the vertices
        V_tmp = hommogenous @ R.T
        
        # Return the transformed vertices
        return V_tmp[:, :3]

    def _bevel_transformation(self, spec: GearSpec, V: np.ndarray, T: np.ndarray, face_width: float) -> tuple[np.ndarray, np.ndarray]:
        """Transforms the gear profile vertices to create a bevel gear.
        
        :param spec: The gear specification.
        :param V: The gear profile vertices.
        :param T: The gear profile triangles.
        :param face_width: The face width of the gear.
        :return: The transformed gear vertices and triangles.
        
        :raises ValueError: If the bevel cone angle is not specified.
        """
        
        if spec.bevel_cone_angle is None:
            raise ValueError("Bevel cone angle must be specified.")
        
        # Compute the radius and height of the pitch cone
        pitch_cone_radius = spec.rp
        pitch_cone_height = pitch_cone_radius / np.tan(spec.bevel_cone_angle)
        
        # Create the hommogenous coordinates and offset the z coordinate
        V_hom = np.hstack((V, np.ones((len(V), 1))))
        V_hom[:, 2] += pitch_cone_height
        
        for i in range(len(V_hom)):
            coords = V_hom[i]

            # Compute the radius of the pitch cone at the current z coordinate
            r_cone = pitch_cone_radius / pitch_cone_height * coords[2]
            
            # Scale x, y coordinates to the radius of the pitch cone
            coords[:2] *= r_cone / spec.rp
            
            V_hom[i] = self._rotate_bevel_coordinates(coords, spec, pitch_cone_height)
        
        V_hom[:, 2] -= pitch_cone_height
        
        return V_hom[:, :3], T
    
    def _rotate_bevel_coordinates(self, coords: np.ndarray, spec: GearSpec, cone_height: float) -> np.ndarray:
        """Rotates the gear profile vertices to create a bevel gear.
        
        :param coords: The gear profile vertices.
        :param spec: The gear specification.
        :param cone_height: The height of the pitch cone.
        :return: The rotated gear profile vertices.
        """
        xy_unit = coords[:2] / np.linalg.norm(coords[:2])
        z = coords[2]
        
        # Compute the radius of the cone at the current z coordinate
        cone_radius = z * spec.rp / cone_height
        # Compute the point on the cone
        cone_point = np.append(cone_radius * xy_unit, z)
        
        # Create the transformation matrices
        # Translation matrix for the cone point
        Mt = GearFactory.__translation_matrix(cone_point)
        # Orientation matrix for the cone point
        Mo = GearFactory.__orientation_matrix(cone_point)
        # Rotation matrix
        Mr = GearFactory.__rotation_matrix_y(-spec.bevel_cone_angle)
        
        # Compute the inverse of the transformation matrices
        Mt_inv = np.linalg.inv(Mt)
        Mo_inv = np.linalg.inv(Mo)
        
        # Compute the final transformation matrix
        M = (
            Mt_inv   # Translate point to origin
            @ Mo_inv # Rotate point to align with axis
            @ Mr     # Rotate point to be perpendicular to cone
            @ Mo     # Restore original orientation
            @ Mt     # Restore original position
        )
        
        return coords @ M

    def __translation_matrix(v: np.ndarray):
        """Compute a translation matrix for the given vector.
        
        :param v: The vector to translate the matrix.
        :return: The translation matrix.
        """
        return np.array([
            [1, 0, 0, v[0]],
            [0, 1, 0, v[1]],
            [0, 0, 1, v[2]],
            [0, 0, 0, 1]
        ]).T

    def __orientation_matrix(v: np.ndarray):
        """Compute an orientation matrix for the given vector.
        
        :param v: The vector to orient the matrix.
        :return: The orientation matrix.
        """
        
        up = np.array([0, 0, 1])
        
        axis_x = v / np.linalg.norm(v)
        axis_y = np.cross(axis_x, up)
        axis_y /= np.linalg.norm(axis_y)
        axis_z = np.cross(axis_x, axis_y)
        axis_z /= np.linalg.norm(axis_z)
        
        return np.array([
            [axis_x[0], axis_y[0], axis_z[0], 0],
            [axis_x[1], axis_y[1], axis_z[1], 0],
            [axis_x[2], axis_y[2], axis_z[2], 0],
            [0, 0, 0, 1]
        ]).T

    def __rotation_matrix_y(theta: float):
        """Compute a rotation matrix around the y-axis.
        
        :param theta: The rotation angle in radians.
        :return: The rotation matrix.
        """
        
        return np.array([
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ]).T
