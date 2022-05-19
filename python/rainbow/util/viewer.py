import numpy as np
import pythreejs as p3js
from IPython.display import display
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
import rainbow.geometry.surface_mesh as MESH


class _ShapeHelper:
    @staticmethod
    def make_mesh_node(V, F, color=None, opacity_value=1.0, wire_frame_on=False):
        F = F.astype(dtype=np.uint32, copy=True).ravel()
        V = V.astype(dtype=np.float32, copy=True)

        attribute_colors = None
        if color is None:
            attribute_colors = np.copy(V)
        elif type(color) == np.ndarray and color.shape[0] == V.shape[0]:
            attribute_colors = np.copy(color)
        elif type(color) == np.ndarray and color.size == 3:  # Single color
            attribute_colors = np.ones_like(V)
            attribute_colors[:, 0] = color[0]
            attribute_colors[:, 1] = color[1]
            attribute_colors[:, 2] = color[2]
        elif len(color) == 3:  # Single color, non-numpy array type
            attribute_colors = np.ones_like(V)
            attribute_colors[:, 0] = color[0]
            attribute_colors[:, 1] = color[1]
            attribute_colors[:, 2] = color[2]

        opacity_value = np.clip(opacity_value, 0.0, 1.0)
        transparency_value = True if opacity_value < 1.0 else False

        attributes = {
            "color": p3js.BufferAttribute(attribute_colors),
            "index": p3js.BufferAttribute(F, normalized=False),
            "position": p3js.BufferAttribute(V, normalized=False),
        }

        geometry = p3js.BufferGeometry(attributes=attributes)
        material = p3js.MeshStandardMaterial(
            vertexColors="VertexColors",
            reflectivity=1.0,
            side="FrontSide",
            roughness=0.5,
            metalness=0.25,
            flatShading=True,
            polygonOffset=True,
            polygonOffsetFactor=1,
            polygonOffsetUnits=5,
            opacity=opacity_value,
            transparent=transparency_value,
        )
        mesh = p3js.Mesh(geometry=geometry, material=material)
        if wire_frame_on:
            wf_geometry = p3js.WireframeGeometry(mesh.geometry)
            wf_material = p3js.LineBasicMaterial(color="#AAAA00", linewidth=0.01)
            wire_frame = p3js.LineSegments(wf_geometry, wf_material)
            mesh.add(wire_frame)

        mesh.geometry.exec_three_obj_method("computeVertexNormals")
        return mesh

    @staticmethod
    def update_mesh_node(mesh, V, wire_frame_on=False):
        V = V.astype(dtype=np.float32, copy=True)
        mesh.geometry.attributes["position"].array = V
        mesh.geometry.attributes["position"].needsUpdate = True
        if wire_frame_on:
            # TODO Kristian: This seems like a really bad way to update the wire-framing
            wf_geometry = p3js.WireframeGeometry(mesh.geometry)
            wf_material = p3js.LineBasicMaterial(color="#AAAA00", linewidth=0.01)
            wire_frame = p3js.LineSegments(wf_geometry, wf_material)
            mesh.children = (wire_frame,)
        mesh.geometry.exec_three_obj_method("computeVertexNormals")

    @staticmethod
    def make_arrow_node(color):
        V, F = MESH.create_sphere(0.25 / 3, 12, 12)
        base = _ShapeHelper.make_mesh_node(V, F, color)

        V, F = MESH.create_cylinder(0.1 / 3, 2.0 / 3, 12)
        shaft = _ShapeHelper.make_mesh_node(V, F, color)
        shaft.position = [0, 1.0 / 3, 0]

        V, F = MESH.create_cone(0.25 / 3, 1.0 / 3, 12)
        cone = _ShapeHelper.make_mesh_node(V, F, color)
        cone.position = [0, 2.0 / 3, 0.0]
        arrow = p3js.Group()
        arrow.add(base)
        arrow.add(shaft)
        arrow.add(cone)
        return arrow

    @staticmethod
    def make_ball_node(color):
        V, F = MESH.create_sphere(0.25 / 3, 12, 12)
        base = _ShapeHelper.make_mesh_node(V, F, color)
        ball = p3js.Group()
        ball.add(base)
        return ball


    @staticmethod
    def make_frame_node():
        V, F = MESH.create_sphere(0.25 / 3, 12, 12)
        base = _ShapeHelper.make_mesh_node(V, F, color = (0.1, 0.1, 0.1))

        #--- y-axis -----------------------------------------
        V, F = MESH.create_cylinder(0.1 / 3, 2.0 / 3, 12)
        yshaft = _ShapeHelper.make_mesh_node(V, F,  color = (0.1, 0.7, 0.1))
        yshaft.position = [0, 1.0 / 3, 0]

        V, F = MESH.create_cone(0.25 / 3, 1.0 / 3, 12)
        ycone = _ShapeHelper.make_mesh_node(V, F,  color = (0.1, 0.7, 0.1))
        ycone.position = [0, 2.0 / 3, 0.0]

        #--- x-axis -----------------------------------------
        Qz = Q.Rz(  -np.pi / 2 )

        V, F = MESH.create_cylinder(0.1 / 3, 2.0 / 3, 12)
        V = Q.rotate_array(Qz, V)
        xshaft = _ShapeHelper.make_mesh_node(V, F,  color = (0.7, 0.1, 0.1))
        xshaft.position = [1.0 / 3, 0.0, 0.0]

        V, F = MESH.create_cone(0.25 / 3, 1.0 / 3, 12)
        V = Q.rotate_array(Qz, V)
        xcone = _ShapeHelper.make_mesh_node(V, F,  color = (0.7, 0.1, 0.1))
        xcone.position = [2.0 / 3, 0.0, 0.0]

        #---  z-axis -----------------------------------------
        Qx = Q.Rx( np.pi / 2 )

        V, F = MESH.create_cylinder(0.1 / 3, 2.0 / 3, 12)
        V = Q.rotate_array(Qx, V)
        zshaft = _ShapeHelper.make_mesh_node(V, F,  color = (0.1, 0.1, 0.8))
        zshaft.position = [0.0, 0.0, 1.0 / 3]

        V, F = MESH.create_cone(0.25 / 3, 1.0 / 3, 12)
        V = Q.rotate_array(Qx, V)
        zcone = _ShapeHelper.make_mesh_node(V, F,  color = (0.1, 0.1, 0.8))
        zcone.position = [0.0, 0.0, 2.0 / 3]

        frame = p3js.Group()
        frame.add(base)
        frame.add(xshaft)
        frame.add(xcone)
        frame.add(yshaft)
        frame.add(ycone)
        frame.add(zshaft)
        frame.add(zcone)
        return frame


class _QuiverHelper:

    @staticmethod
    def create_quiver_node(V, N, scale=1.0, color=(0.1, 0.1, 0.7)):
        quiver = p3js.Group()
        if V is not None:
            quiver.visible = False
            K = len(V)
            for k in range(K):
                # TODO 2020-12-20 Kenny: Geometry instancing would be nice
                arrow = _ShapeHelper.make_arrow_node(color)
                arrow.visible = True
                arrow.position = [V[k, 0], V[k, 1], V[k, 2]]
                # By default the arrow geometry points along the positive y-axis. We must compute the
                # rotation so the arrow points along the N-axis instead.
                n = V3.unit(V3.make(N[k, 0], N[k, 1], N[k, 2]))

                r = V3.cross(V3.j(), n)
                if np.any(r):
                    rad = np.arccos(np.clip(np.dot(V3.j(), n), -1.0, 1.0))
                    q = Q.Ru(rad, r)
                elif  n.dot(V3.j()) < 0:
                    q = Q.Rx(np.pi)
                arrow.quaternion = [q[1], q[2], q[3], q[0]]

                magnitude = V3.norm(N)*scale
                arrow.scale = [magnitude, magnitude, magnitude]
                quiver.add(arrow)
            quiver.visible = True
        return quiver

    @staticmethod
    def update_quiver_node(quiver, V, N, scale=1.0, color=(0.1, 0.1, 0.7)):
        quiver.visibility = False
        # TODO 2021-04-20 Kenny: This does not quite protect against rendering artefact. Sometimes one can se dangling
        #  arrows which do not really exist. There need to be some kind of double buffering to make this work.
        K = len(V)
        # Create more arrows if we have too many contact points.
        while len(quiver.children) < K:
            # TODO 2020-12-20 Kenny: Geometry instancing would be nice
            arrow = _ShapeHelper.make_arrow_node(color)
            quiver.add(arrow)
        k = 0
        for arrow in quiver.children:
            if k < K:
                arrow.visible = True
                arrow.position = [V[k, 0], V[k, 1], V[k, 2]]
                # By default the arrow geometry points along the positive y-axis. We must compute the
                # rotation so the arrow points along the N-axis instead.
                n = V3.unit(V3.make(N[k, 0], N[k, 1], N[k, 2]))

                r = V3.cross(V3.j(), n)
                if np.any(r):
                    rad = np.arccos(np.clip(np.dot(V3.j(), n), -1.0, 1.0))
                    q = Q.Ru(rad, r)
                elif  n.dot(V3.j()) < 0:
                    q = Q.Rx(np.pi)
                arrow.quaternion = [q[1], q[2], q[3], q[0]]

                magnitude = V3.norm(N)*scale
                arrow.scale = [magnitude, magnitude, magnitude]
            else:
                arrow.visible = False
            k = k + 1
        quiver.visible = True


class _ScatterHelper:

    @staticmethod
    def create_scatter_node(V, scale=1.0, color=(0.1, 0.1, 0.7)):
        scatter = p3js.Group()
        if V is not None:
            quiver.visible = False
            K = len(V)
            for k in range(K):
                # TODO 2020-12-20 Kenny: Geometry instancing would be nice
                ball = _ShapeHelper.make_ball_node(color)
                ball.visible = True
                ball.position = [V[k, 0], V[k, 1], V[k, 2]]
                ball.scale = [scale, scale, scale]
                scatter.add(ball)
            scatter.visible = True
        return scatter

    @staticmethod
    def update_scatter_node(scatter, V, scale=1.0, color=(0.1, 0.1, 0.7)):
        scatter.visibility = False
        # TODO 2021-04-20 Kenny: This does not quite protect against rendering artefact. Sometimes one can se dangling
        #  arrows which do not really exist. There need to be some kind of double buffering to make this work.
        K = len(V)
        # Create more arrows if we have too many contact points.
        while len(scatter.children) < K:
            # TODO 2020-12-20 Kenny: Geometry instancing would be nice
            ball = _ShapeHelper.make_ball_node(color)
            scatter.add(ball)
        k = 0
        for ball in scatter.children:
            if k < K:
                ball.visible = True
                ball.position = [V[k, 0], V[k, 1], V[k, 2]]
                ball.scale = [scale, scale, scale]
            else:
                ball.visible = False
            k = k + 1
        scatter.visible = True


class Viewer:
    
    def __init__(self, width=600, height=480):
        self.scatters = {}
        self.quivers = {}
        self.meshes = {}
        self.width = int(width)
        self.height = int(height)
        self.aspect = 1.0 * self.width / self.height
        self.fov = 30.0
        self.flash_light = p3js.DirectionalLight(
            color="white", position=[0, 0, 1], intensity=0.6
        )
        self.sun_light = p3js.AmbientLight(intensity=0.5)
        self.camera = p3js.PerspectiveCamera(
            position=[0, 0, 20],
            lookAt=[0, 0, 0],
            fov=self.fov,
            aspect=self.aspect,
            children=[self.flash_light],
        )
        self.orbit = p3js.OrbitControls(controlling=self.camera)
        self.scene = p3js.Scene(
            children=[self.camera, self.sun_light], background="#FFFFFF"
        )
        self.renderer = p3js.Renderer(
            camera=self.camera,
            scene=self.scene,
            controls=[self.orbit],
            width=self.width,
            height=self.height,
            antialias=True,
        )

    def show(self):
        display(self.renderer)

    def create_mesh(self, name, V, T, color=None, opacity_value=1.0, wire_frame_on=True):
        if name in self.meshes:
            raise ValueError("Mesh with that name already exists")
        mesh_node = _ShapeHelper.make_mesh_node(
            V, T, color, opacity_value, wire_frame_on
        )
        self.meshes[name] = mesh_node
        self.scene.add(mesh_node)

    def place_mesh(self, name, r, q):
        if name not in self.meshes:
            raise ValueError("Mesh with that name did not exists")
        obj = self.meshes[name]
        position = [r[0], r[1], r[2]]
        orientation = [q[1], q[2], q[3], q[0]]
        obj.position = position
        obj.quaternion = orientation

    def update_mesh(self, name, V, wire_frame_on=True):
        if name not in self.meshes:
            raise ValueError("Mesh with that name did not exists")
        mesh = self.meshes[name]
        _ShapeHelper.update_mesh_node(mesh, V, wire_frame_on)

    def create_quiver(self, name, V=None, N=None, scale=1.0, color=(0.1, 0.1, 0.7)):
        if name in self.quivers:
            raise ValueError("Quiver with that name already exists")
        quiver = _QuiverHelper.create_quiver_node(V, N, scale, color)
        self.quivers[name] = quiver
        self.scene.add(quiver)

    def hide_quiver(self, name):
        if name not in self.quivers:
            raise ValueError("No quiver exist with that name")
        quiver = self.quivers[name]
        quiver.visible = False

    def show_quiver(self, name):
        if name not in self.quivers:
            raise ValueError("No quiver exist with that name")
        quiver = self.quivers[name]
        quiver.visible = True

    def update_quiver(self, name, V, N, scale=1.0, color=(0.1, 0.1, 0.7)):
        if name not in self.quivers:
            raise ValueError("No quiver exist with that name")
        quiver = self.quivers[name]
        _QuiverHelper.update_quiver_node(quiver, V, N, scale, color)

    def create_scatter(self, name, V=None, N=None, scale=1.0, color=(0.1, 0.1, 0.7)):
        if name in self.scatters:
            raise ValueError("Scatter with that name already exists")
        scatter = _ScatterHelper.create_scatter_node(V, scale, color)
        self.scatters[name] = scatter
        self.scene.add(scatter)

    def hide_scatter(self, name):
        if name not in self.scatters:
            raise ValueError("No scatter exist with that name")
        scatter = self.scatters[name]
        scatter.visible = False

    def show_scatter(self, name):
        if name not in self.scatters:
            raise ValueError("No scatter exist with that name")
        scatter = self.scatters[name]
        scatter.visible = True

    def update_scatter(self, name, V, scale=1.0, color=(0.1, 0.1, 0.7)):
        if name not in self.scatters:
            raise ValueError("No scatter exist with that name")
        scatter = self.scatters[name]
        _ScatterHelper.update_scatter_node(scatter, V, scale, color)

    def create_frame(self, name):
        if name in self.meshes:
            raise ValueError("Mesh with that name already exists")
        mesh_node = _ShapeHelper.make_frame_node()
        self.meshes[name] = mesh_node
        self.scene.add(mesh_node)

    def place_frame(self, name, r, q, scale=1.0):
        if name not in self.meshes:
            raise ValueError("Mesh with that name did not exists")
        obj = self.meshes[name]
        position = [r[0], r[1], r[2]]
        orientation = [q[1], q[2], q[3], q[0]]
        obj.position = position
        obj.quaternion = orientation
        obj.scale = scale
