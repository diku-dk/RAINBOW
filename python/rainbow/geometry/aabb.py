import numpy as np
from numpy.typing import ArrayLike


class AABB:
    """
    Axis-Aligned Bounding Box (AABB) for 3D spatial objects.

    An AABB is a box that encloses a 3D object, aligned with the coordinate axes. 
    It is defined by two points: the minimum point (`min_point`) and the maximum point (`max_point`),
    which represent opposite corners of the box.

    Attributes:
        min_point (ArrayLike): The smallest x, y, z coordinates from the bounding box.
        max_point (ArrayLike): The largest x, y, z coordinates from the bounding box.

    Methods:
        create_from_vertices(vertices: ArrayLike) -> 'AABB'
            Class method to create an AABB instance from a list of vertices.

        is_overlap(aabb1: 'AABB', aabb2: 'AABB') -> bool
            Class method to determine if two AABB instances overlap.

    Example:
        >>> aabb1 = AABB([0, 0, 0], [1, 1, 1])
        >>> vertices = [[0, 0, 0], [1, 1, 1], [1, 0, 0]]
        >>> aabb2 = AABB.create_from_vertices(vertices)
        >>> AABB.is_overlap(aabb1, aabb2)
        True
    """
    def __init__(self, min_point: ArrayLike, max_point: ArrayLike) -> None:
        self.min_point = np.array(min_point, dtype=np.float64)
        self.max_point = np.array(max_point, dtype=np.float64)
        
    @classmethod
    def create_from_vertices(cls, vertices: ArrayLike) -> 'AABB':
        """ Create AABB instance from vertices, such as triangle vertices

        Args:
            vertices (List[List[float]]): A list of vertices, each vertex is a list of 3 elements

        Returns:
            AABB: a new AABB instance
        """
        max_point = np.max(vertices, axis=0)
        min_point = np.min(vertices, axis=0)
        return cls(min_point, max_point)
    
    @classmethod
    def is_overlap(cls, aabb1: 'AABB', aabb2: 'AABB', boundary: float = 0.0) -> bool:
        """ Test two aabb instance are overlap or not

        Args:
            aabb1 (AABB): The AABB instance of one object 
            aabb2 (AABB): The AABB instance of one object 
            boundary (float): which is used to expand the aabb, hence we should use a positive floating point, Defaults to 0.0.

        Returns:
            bool: Return True if both of aabb instances are overlap, otherwise return False
        """
        if boundary != 0.0:
            aabb1_min_copy = np.copy(aabb1.min_point)
            aabb1_max_copy = np.copy(aabb1.max_point)
            aabb2_min_copy = np.copy(aabb2.min_point)
            aabb2_max_copy = np.copy(aabb2.max_point)
            aabb1_min_copy -= boundary
            aabb1_max_copy += boundary
            aabb2_min_copy -= boundary
            aabb2_max_copy += boundary
            return not (np.any(aabb1_max_copy < aabb2_min_copy) or np.any(aabb1_min_copy > aabb2_max_copy))
        else:
            return not (np.any(aabb1.max_point < aabb2.min_point) or np.any(aabb1.min_point > aabb2.max_point))