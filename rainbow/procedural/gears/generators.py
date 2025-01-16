import numpy as np

import rainbow.math.involute as INVOLUTE

from .types import GearSpec


class GearFactory:
    def __init__(self, side_points: int = 10, top_points: int = 3, bottom_points: int = 3) -> None:
        self.side_points = side_points
        self.top_points = top_points
        self.bottom_points = bottom_points
    
    def create_involute_gear_mesh(self, spec: GearSpec, face_width: float) -> tuple[np.ndarray, np.ndarray]:
        V_profile = self._create_profile_points(spec)
        V_cylinder = self._create_cylinder_points(spec)
        
        V, T = self._tessellation(spec, V_profile, V_cylinder)
        V, T = self._extrude(spec, V, T, face_width)
        
        return V, T

    def points_per_tooth(self, spec: GearSpec) -> int:
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
        return spec.z * self.points_per_tooth(spec)

    def _create_profile_points(self, spec: GearSpec) -> np.ndarray:
        ts = np.linspace(spec.t_min, spec.t_max, self.side_points)
        rev_ts = -ts[::-1]
        
        xs = []
        ys = []
        for idx in range(spec.z):
            theta_i = spec.theta_i[idx]
            theta_i_next = spec.theta_i[(idx + 1) % spec.z]
            if theta_i_next < theta_i:
                theta_i_next += 2 * np.pi
            
            right_root_offset = theta_i - spec.gamma_b / 2
            left_root_offset = theta_i + spec.gamma_b / 2
            
            # Right side
            right_inv_x = INVOLUTE.create_involute_x_function(spec.rb, right_root_offset)
            right_inv_y = INVOLUTE.create_involute_y_function(spec.rb, right_root_offset)
            xs.extend(right_inv_x(ts))
            ys.extend(right_inv_y(ts))
            
            # Top
            right_top_offset = theta_i - spec.gamma_a / 2
            left_top_offset = theta_i + spec.gamma_a / 2
            top_theta = np.linspace(right_top_offset, left_top_offset, self.top_points + 2)[1:-1]
            xs.extend(spec.ra * np.cos(top_theta))
            ys.extend(spec.ra * np.sin(top_theta))
            
            # Left side
            left_inv_x = INVOLUTE.create_involute_x_function(spec.rb, left_root_offset)
            left_inv_y = INVOLUTE.create_involute_y_function(spec.rb, left_root_offset)
            xs.extend(left_inv_x(rev_ts))
            ys.extend(left_inv_y(rev_ts))
            
            # Bottom
            if spec.rb >= spec.rd:
                bottom_theta = np.linspace(left_root_offset, left_root_offset + 2 * spec.gamma_p - spec.gamma_b, self.bottom_points + 2)
                xs.extend(spec.rd * np.cos(bottom_theta))
                ys.extend(spec.rd * np.sin(bottom_theta))
            else:
                bottom_start = theta_i + spec.gamma_d / 2
                bottom_end = theta_i_next - spec.gamma_d / 2
                bottom_theta = np.linspace(bottom_start, bottom_end, self.bottom_points + 2)[1:-1]
                xs.extend(spec.rd * np.cos(bottom_theta))
                ys.extend(spec.rd * np.sin(bottom_theta))
        
        return np.vstack((xs, ys, np.zeros_like(xs))).T

    def _create_cylinder_points(self, spec: GearSpec) -> np.ndarray:
        r = 1.2 * spec.ra if spec.is_internal else 0.2 * spec.rp
        theta = np.linspace(0, 2 * np.pi, spec.z + 1)[:-1]
        if spec.is_internal:
            theta += np.pi / spec.z
        
        xs = r * np.cos(theta)
        ys = r * np.sin(theta)
        
        return np.vstack((xs, ys, np.zeros_like(xs))).T

    def _tessellation(self, spec: GearSpec, V_profile: np.ndarray, V_cylinder: np.ndarray) -> np.ndarray:
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

    def _extrude(self, spec: GearSpec, V: np.ndarray, T: np.ndarray, face_width: float) -> np.ndarray:
        total_gear_points = self.total_gear_points(spec)
        
        V1 = np.copy(V)
        V2 = np.copy(V)
        V2[:, 2] = face_width
        
        T1 = np.copy(T)
        T2 = T[:, ::-1] + len(V)
        T_profile = []
        for i in range(total_gear_points):
            j1 = i + V1.shape[0]
            j2 = (i + 1) % total_gear_points
            k = ((i + 1) % total_gear_points) + V1.shape[0]
            T_profile.append((i, k, j1))
            T_profile.append((i, j2, k))

        T_cylinder = []
        for idx in range(spec.z):
            i = idx + total_gear_points
            j1 = i + V1.shape[0]
            j2 = (idx + 1) % spec.z + total_gear_points
            k = ((idx + 1) % spec.z) + total_gear_points + V1.shape[0]
            T_cylinder.append((i, j1, k))
            T_cylinder.append((i, k, j2))
        
        V_final = np.vstack((V1, V2))
        V_final[:, 2] -= face_width / 2
        
        T_final = np.vstack((T1, T2, T_profile, T_cylinder))
        if spec.is_internal:
            T_final = T_final[:,::-1]
        
        return V_final, T_final
