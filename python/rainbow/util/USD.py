from pxr import Usd, UsdGeom, Gf, Vt
import numpy as np

class USD:
    """
        USD: Universal Scene Description, which is a file format for 3D data (https://graphics.pixar.com/usd/docs/index.html).
        
        Class for record the simulation processing to USD format , then writing it to a USD file.
        
        TODO: We only convert the position changing of the mesh, does not support to record the Camrea,Lights,Color,etc.
    """

    def __init__(self, file_path):
        self.stage = Usd.Stage.CreateInMemory(file_path)
        self.xform = UsdGeom.Xform.Define(self.stage, '/scene/xform')
        self.meshes = dict()
        self.file_path = file_path

    def add_mesh(self, name, V, T):
        mesh = UsdGeom.Mesh.Define(self.stage, '/scene/xform/{}'.format(name))
        mesh.CreatePointsAttr(V)
        mesh.CreateFaceVertexCountsAttr([len(face) for face in T])
        mesh.CreateFaceVertexIndicesAttr(np.concatenate(T))
        self.meshes[name] = mesh
    
    def update_mesh_positions(self, name, V, time):
        if name not in self.meshes:
            raise ValueError('Mesh {} does not exist'.format(name))
        vertex_positions = Vt.Vec3fArray(V.tolist())
        self.meshes[name].GetPointsAttr().Set(vertex_positions, time)

    def set_animation_time(self, duration):
        self.stage.SetStartTimeCode(0)
        self.stage.SetEndTimeCode(duration)

    def save(self):
        self.stage.GetRootLayer().Export(self.file_path)

    

