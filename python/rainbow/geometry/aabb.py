import numpy as np
from typing import List

class AABB:
    """
    Axis-Aligned Bounding Box (AABB) for 3D spatial objects.

    An AABB is a box that encloses a 3D object, aligned with the coordinate axes. 
    It is defined by two points: the minimum point (`min_point`) and the maximum point (`max_point`),
    which represent opposite corners of the box.

    Attributes:
        min_point (List[float]): The smallest x, y, z coordinates from the bounding box.
        max_point (List[float]): The largest x, y, z coordinates from the bounding box.

    Methods:
        create_from_vertices(vertices: List[List[float]]) -> 'AABB'
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
    
    def __init__(self, min_point: List[float], max_point: List[float]) -> None:
        self.min_point = min_point
        self.max_point = max_point
        
    @classmethod
    def create_from_vertices(cls, vertices: List[List[float]]) -> 'AABB':
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
    def is_overlap(cls, aabb1: 'AABB', aabb2: 'AABB') -> bool:
        """ Test two aabb instance are overlap or not

        Args:
            aabb1 (AABB): _description_
            aabb2 (AABB): _description_

        Returns:
            bool: Return True if both of aabb instances are overlap, otherwise return False
        """
        return not (aabb1.max_point[0] < aabb2.min_point[0] or aabb1.min_point[0] > aabb2.max_point[0] or
                    aabb1.max_point[1] < aabb2.min_point[1] or aabb1.min_point[1] > aabb2.max_point[1] or
                    aabb1.max_point[2] < aabb2.min_point[2] or aabb1.min_point[2] > aabb2.max_point[2])