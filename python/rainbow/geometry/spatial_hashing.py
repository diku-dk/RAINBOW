import numpy as np
from typing import Any, List, Tuple
from rainbow.geometry.aabb import AABB


class HashCell:
    """
    A class representing a hash cell in spatial hashing for quick lookup and 
    managing spatial-related objects, such as triangles in a 3D mesh.

    The `HashCell` class allows for efficient management of objects (such as 
    triangles in a mesh) within a spatial hashing grid. It uses a "lazy clear" 
    mechanism, resetting the cell only when a new object is added with a more 
    recent timestamp, to optimize object management in dynamic simulations.
    
    Attributes:
        time_stamp (int): A marker representing the last moment when the 
            cell was accessed or modified.
        size (int): The number of objects currently stored in the cell.
        object_list (List[Any]): A list holding the objects stored in the cell.

    Methods:
        add(object: Any, time_stamp: int)
            Adds an object to the cell and updates the time stamp, 
            performing a lazy clear if needed.

    Example:
        >>> cell = HashCell()
        >>> cell.add(("triangle", "body_name", "aabb"), 1)
        >>> cell.size
        1
    
    Note:
        The objects stored can be of any type (`Any`), but for applications 
        like collision detection, it is recommended to store relevant spatial 
        data, such as a tuple containing (triangle index, body name, triangle AABB).
    """
    def __init__(self, time_stamp: int=0) -> None:
        self.time_stamp = time_stamp
        self.size = 0
        self.object_list = []

    def add(self, object: Any, time_stamp: int):
        """ Add an object to the cell

        Args:
            object (Any): This object can be a triangle index, a body name, or a triangle AABB, it depends on the context. In the context of collision detection, it is a tuple of (triangle index, body name, triangle AABB)
            time_stamp (int): The time stamp of the simulation program
        """
        # Lazy Clear: If the time stamp is older than the current time stamp, reset the cell
        if self.time_stamp < time_stamp:
            self.time_stamp = time_stamp
            self.size = 0
            self.object_list = []
        
        self.object_list.append(object)
        self.size += 1


class HashGird:
    """
    A class representing a 3D spatial hash grid for efficient spatial 
    querying and management of objects, such as triangles in a 3D mesh.

    The `HashGrid` uses a hash function to map spatial cells into a 1D 
    hash table, which allows for an efficient query of neighboring objects 
    in a spatial domain, commonly used in collision detection and other 
    physical simulations.

    Attributes:
        hash_table_size (int): Size of the hash table, dictating how many 
            possible hashed keys/values pairs it can manage.
        hash_table (dict): Dictionary acting as the hash table, 
            storing objects in the spatial grid.
        cell_size (np.array): 1D numpy array containing the 3D dimensions 
            of a cell in the grid (x, y, z).

        offset_table_size(int): The size of the offset table(Phi)
        M0(Identity Matrix): A linear transfomation matrix used to map the domain(U) to the hash tbale(H)
        M1(Identity Matrix): A linear transfomation matrix used to map the domain(U) to the offset table(Phi)
        Phi(List[[int, int, int]]): The offset table used to remove the collision of the hash function

    Methods:
        set_hash_table_size(hash_table_size: int)
            Sets the size of the hash table
        increment_hash_table_size(increment_size: int)
            Increments the size of the hash table
        set_cell_size(cell_size_x: float, cell_size_y: float, cell_size_z: float)
            Sets the 3D dimensions of a cell in the grid.
        get_hash_value(i: int, j: int, k: int) -> int
            Computes and returns the hash value for a spatial cell given 
            its 3D grid indices (i, j, k).
        insert(i: int, j: int, k: int, tri_idx: int, body_name: str, 
               tri_aabb: AABB, time_stamp: int) -> list
            Inserts a triangle into the hash grid and returns a list of 
            objects in the cell, performing collision checks.

    Example:
        >>> hash_grid = HashGrid()
        >>> hash_grid.set_cell_size(1.0, 1.0, 1.0)
        >>> hash_grid.insert(1, 2, 3, 0, "body1", aabb, 1)

    Note:
        The objects inserted into the `HashGrid` are typically related 
        to spatial entities (such as triangles in a 3D mesh) and include 
        details like an index, body name, and an axis-aligned bounding box (AABB).
    """

    def __init__(self) -> None:
        self.hash_table_size = 1000
        self.hash_tbale = dict()
        self.cell_size = 0.0
        
        # Perfect Hashing Setup: These parameters are configured and subsequently used in the get_prefect_hash_value function.
        self.offset_table_size = 1000
        self.M0 = np.eye(3, dtype=int)
        self.M1 = np.eye(3, dtype=int)
        self.Phi = np.random.randint(self.hash_table_size, size=(self.offset_table_size,) * 3)
        self.mod_value = 1e9 + 7
    
    def set_hash_table_size(self, hash_table_size: int):
        """ Set the size of the hash table

        Args:
            hash_table_size (int32): The size of the hash table
        """
        self.hash_table_size = hash_table_size
    
    def increment_hash_table_size(self, increment_size: int):
        """ Increment the size of the hash table

        Args:
            increment_size (int32): The size of the hash table
        """
        self.hash_table_size = self.hash_table_size + increment_size

    def set_cell_size(self, cell_size: float):
        """ Set the x, y, z axis length of a cell

        Args:
            cell_size (float): the cell size 
        """
        self.cell_size = cell_size
    
    def get_prefect_hash_value(self, i: int, j: int, k: int) -> int:
        """ Get the prefect hash value of the cell.
            The hash function h(p) is computed as follows:
            h(p) = (h_0(p) + Phi(h_1(p))) % m
            where:
            h_0(p): is the primary hash function used to calculate the hash value,
            h_1(p): is a secondary hash function used to calculate the offset,
            Phi: is the offset table,
            m: is the size of the hash table.
            For more information, refer to the 3rd section of this paper: https://dl.acm.org/doi/10.1145/1141911.1141926 

        Args:
            i (int): The i index of the cell of X axis
            j (int): The j index of the cell of Y axis
            k (int): The k index of the cell of Z axis

        Returns:
            int: The prefect hash value of the cell
        """
        p = np.array([i, j, k])
        h0 = np.dot(p, self.M0) % self.hash_table_size
        h1 = np.dot(p, self.M1) % self.offset_table_size
        hv = (h0 + self.Phi[tuple(h1)]) % self.hash_table_size

        return int(np.sum(hv) % self.mod_value)

    def insert(self, i: int, j: int, k: int, tri_idx: int, body_idx: int, tri_aabb: 'AABB', time_stamp: int) -> list:
        """ Insert a triangle into the hash table, and return the list of object of the cell

        Args:
            i (int): The i index of the cell of X axis
            j (int): The j index of the cell of Y axis
            k (int): The k index of the cell of Z axis
            tri_idx (int): The index of the triangle of the body
            body_idx (int): The index of the body
            tri_aabb (AABB): The AABB of the triangle
            time_stamp (int): The time stamp of the simulation program

        Returns:
            list: The list of object of the cell
        """
        overlaps = []
        hv = self.get_prefect_hash_value(i, j, k)
        if hv not in self.hash_tbale:
            self.hash_tbale[hv] = HashCell()
            self.hash_tbale[hv].add((tri_idx, body_idx, tri_aabb), time_stamp)
        else:
            overlaps = self.hash_tbale[hv].object_list
            self.hash_tbale[hv].add((tri_idx, body_idx, tri_aabb), time_stamp)
        return overlaps
    
    @classmethod
    def compute_optial_cell_size(cls, V, T):
        """ Aim to compute the optimal cell size for the spatial hashing, which is the average edge length of the mesh

        Args:
            V (list): The vertices of the mesh
            T (list): The triangles of the mesh

        Returns:
            float: The optimal cell size : 2.2 * average edge length
        """
        edges = []
        for t in T:
            edges.append(V[t[1]] - V[t[0]])
            edges.append(V[t[2]] - V[t[1]])
            edges.append(V[t[0]] - V[t[2]])
        edges = np.array(edges)
        edge_lengths = np.linalg.norm(edges, axis=1)
        
        # the optimal cell size is 2.2 times the average edge length of the surface mesh by our experiments
        return np.mean(edge_lengths) * 2.2