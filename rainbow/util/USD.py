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

    def add_rigid_body(self, name: str, V: ArrayLike, T: ArrayLike) -> None:
        """ Add a rigid body to the scene.
        
        Args:
            name (str): The name of the rigid body.
            V (ArrayLike): The vertex positions of the mesh.
            T (ArrayLike): The triangle faces of the mesh.
        """
        assert name not in self.meshes
        self.meshes[name] = _XFormableWrapper(self.stage, name, V, T)

    def update_rigid_body(self, name: str, translation: ArrayLike, orientation: ArrayLike, time: float) -> None:
        """ Update the translation and orientation of a rigid body.
        
        Args:
            name (str): The name of the rigid body.
            translation (ArrayLike): The translation of the rigid body.
            orientation (ArrayLike): The orientation (a quaternion) of the rigid body.
            time (float): The timestamp for the translation and orientation.

        Raises:
            ValueError: If the rigid body does not exist in the scene.
            TypeError: If the type of the rigid body is incorrect. This will be thrown if the rigid body was added with `add_mesh`.
        """
        xformable = self.meshes.get(name)
        if xformable is None:
            raise ValueError(f'Rigid body {name} does not exist')
        
        if not isinstance(xformable, _XFormableWrapper):
            raise TypeError(f'Rigid body {name} has the wrong type.')

        assert len(translation) == 3
        assert len(orientation) == 4

        xformable.set_translation(translation, time)
        xformable.set_orientation(orientation, time)

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

    def set_animation_time(self, duration: int) -> None:
        """ Set the total animation time of the scene

        Args:
            duration (int): The total animation time of the scene
        """
        print(f'End time code: {duration}')
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(int(duration))

    def set_frames_per_second(self, fps: float):
        self.stage.SetFramesPerSecond(fps)

    def save(self) -> None:
        """ Save the scene to a USD file
        """
        self.stage.GetRootLayer().Export(self.file_path)


class _XFormableWrapper:
    def __init__(self, stage: Usd.Stage, name: str, V: ArrayLike, T: ArrayLike):
        """ Initializes an Xformable in the given stage.

        Args:
            stage (Usd.Stage): The stage the Xformable should be added to.
            name (str): The name of the Xformable.
            V (ArrayLike): The vertex positions of the Xformable.
            T (ArrayLike): The triangle faces of the Xformable.
        """
        self.name = name

        path = f'/scene/xform/{name}'
        mesh = UsdGeom.Mesh.Define(stage, path)        
        mesh.GetPointsAttr().Set(V)
        mesh.GetFaceVertexIndicesAttr().Set(T)
        mesh.GetFaceVertexCountsAttr().Set([3] * len(T))
        
        self.xformable = UsdGeom.Xformable(mesh)
    
    def set_translation(self, translation: ArrayLike, time: float):
        """ Set the translation at the given time.
        
        Args:
            translation (ArrayLike): Translation of the Xformable.
            time (float): The timestamp for the translation.
        """
        assert len(translation) == 3
        translateOp = self._get_xform_op(UsdGeom.XformOp.TypeTranslate)
        translateOp.Set(Gf.Vec3d(*translation), time=time)

    def set_orientation(self, orientation: ArrayLike, time: float):
        """ Set the orientation at the given time.
        
        Args:
            orientation (ArrayLike): Orientation of the Xformable.
            time (float): The timestamp for the orientation.
        """
        assert len(orientation) == 4
        orientOp = self._get_xform_op(UsdGeom.XformOp.TypeOrient)
        orientOp.Set(Gf.Quatd(*orientation), time=time)

    def _get_xform_op(self, xformOpType: UsdGeom.XformOp):
        """ Get or add an XformOp of the given type.
        
        Args:
            xformOpType (UsdGeom.XformOp): The type of transformation.
        """
        ops = self.xformable.GetOrderedXformOps()
        for op in ops:
            if op.GetOpType() == xformOpType:
                return op
        return self.xformable.AddXformOp(xformOpType)
