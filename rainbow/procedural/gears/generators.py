import numpy as np

import rainbow.math.involute as INVOLUTE

from .types import GearSpec


class GearFactory:
    def __init__(self, side_points: int = 10, top_points: int = 3, bottom_points: int = 3) -> None:
        """Creates a new gear factory.
        
        :param side_points: The number of side points for the gear.
        :param top_points: The number of top points for the gear.
        :param bottom_points: The number of bottom points for the gear.
        """
        
        self.side_points = side_points
        self.top_points = top_points
        self.bottom_points = bottom_points
    
    def create_involute_gear_mesh(self, spec: GearSpec, face_width: float, subdivisions: int = 3) -> tuple[np.ndarray, np.ndarray]:
        """Creates the mesh for an involute gear.
        
        :param spec: The gear specification.
        :param face_width: The face width of the gear.
        :param subdivisions: The number of subdivisions for the face width.
        :return: The gear vertices and triangles.
        """
        
        V_profile = self._create_profile_points(spec)
        V_cylinder = self._create_cylinder_points(spec)
        
        V, T = self._tessellation(spec, V_profile, V_cylinder)
        V, T = self._extrude(spec, V, T, face_width, subdivisions)
        
        return V, T

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
        
        r = 1.2 * spec.ra if spec.is_internal else 0.2 * spec.rp
        theta = np.linspace(0, 2 * np.pi, spec.z + 1)[:-1]
        if spec.is_internal:
            theta += np.pi / spec.z
        
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
        
        return np.vstack((xs, ys, np.zeros_like(xs))).T

    def _tessellation(self, spec: GearSpec, V_profile: np.ndarray, V_cylinder: np.ndarray) -> np.ndarray:
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

        index_shift = (points_per_tooth - self.top_points) // 2 if spec.is_internal else 0
        profile_indices = np.array([[i + j * points_per_tooth for i in range(points_per_tooth)] for j in range(spec.z)])
        profile_indices = (profile_indices + index_shift) % total_gear_points
        circle_indices = np.array(range(total_gear_points, total_gear_points + spec.z))
        
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
        
        # Create a copy and update the z coordinate
        V_tmp = np.copy(V)
        V_tmp[:, 2] = (i + 1) * face_width_step
        
        # Rotate the profile if necessary
        if spec.beta is not None and spec.beta != 0:
            # Create the rotation matrix
            theta = spec.beta * (i + 1) / (subdivisions + 1)
            R = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        
            # Transform the vertices
            V_tmp = V_tmp @ R.T
        
        return V_tmp
