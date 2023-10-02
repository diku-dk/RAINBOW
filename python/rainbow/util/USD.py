from pxr import Usd, UsdGeom, Gf, Vt
import numpy as np
from numpy.typing import ArrayLike


class USD:
    """
    Represents a Universal Scene Description (USD) for 3D data.

    Universal Scene Description (USD) is a file format developed by Pixar for representing 3D data. 
    This class facilitates recording simulation processes in USD format and subsequently writing to a USD file.

    Note:
        Currently, the class supports only the conversion of position changes in the mesh. Advanced features such as
        camera, lights, color, etc. are not yet supported.

    Attributes:
        stage (Usd.Stage): The primary container for composing and interrogating scene description.
        xform (UsdGeom.Xform): A transform applied to the scene.
        meshes (dict): A dictionary of meshes added to the scene, with mesh names as keys.
        file_path (str): Path to save the USD file.

    Example:
        >>> usd_instance = USD("path/to/save/file.usd")
        >>> usd_instance.add_mesh("sample_mesh", vertex_positions, triangle_faces)
        >>> usd_instance.set_mesh_positions("sample_mesh", new_positions, time_stamp)
        >>> usd_instance.save()

    See Also:
        [Universal Scene Description (USD) Docs](https://graphics.pixar.com/usd/docs/index.html)
    """

    def __init__(self, file_path: str):
        self.stage = Usd.Stage.CreateInMemory(file_path)
        self.xform = UsdGeom.Xform.Define(self.stage, '/scene/xform')
        self.meshes = dict()
        self.file_path = file_path

    def add_mesh(self, name: str, V: ArrayLike, T: ArrayLike) -> None:
        """ Add a mesh to the scene(or called the stage in USD)

        Args:
            name (str): The name of the mesh
            V (ArrayLike): The vertex positions of the mesh
            T (ArrayLike): The triangle faces of the mesh
        """
        mesh = UsdGeom.Mesh.Define(self.stage, f'/scene/xform/{name}')
        mesh.CreatePointsAttr(V)
        mesh.CreateFaceVertexCountsAttr([len(face) for face in T])
        mesh.CreateFaceVertexIndicesAttr(np.concatenate(T))
        self.meshes[name] = mesh

    def set_mesh_positions(self, name: str, V: ArrayLike, time: float) -> None:
        """_summary_

        Args:
            name (str): The name of the mesh
            V (ArrayLike): The vertex positions of the mesh
            time (float): The timestamp when the mesh is positioned at V

        Raises:
            ValueError: If the mesh does not exist in the scene
        """
        if name not in self.meshes:
            raise ValueError(f'Mesh {name} does not exist')
        vertex_positions = Vt.Vec3fArray(V.tolist())
        self.meshes[name].GetPointsAttr().Set(vertex_positions, time)
    
    def get_mesh_positions(self, name: str, time: float) -> ArrayLike:
        """ Retrieve the positions of a mesh at a given timestamp.

        Args:
            name (str): The name of the mesh.
            time (float): The timestamp at which the positions should be retrieved.

        Returns:
            ArrayLike: An array containing the vertex positions of the mesh.

        Raises:
            ValueError: If the mesh does not exist in the scene, or if the mesh does not have positions set at the given timestamp.
        """
        if name not in self.meshes:
            raise ValueError(f'Mesh {name} does not exist')
        
        vertex_positions_attr = self.meshes[name].GetPointsAttr()
        vertex_positions = vertex_positions_attr.Get(time)

        if vertex_positions:
            return np.array(vertex_positions, dtype=np.float64)
        else:
            raise ValueError(f"No positions set for mesh {name} at time {time}")

    def set_animation_time(self, duration: float) -> None:
        """ Set the total animation time of the scene

        Args:
            duration (float): The total animation time of the scene
        """
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(duration)

    def save(self) -> None:
        """ Save the scene to a USD file
        """
        self.stage.GetRootLayer().Export(self.file_path)