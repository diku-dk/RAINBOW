import numpy as np
import polyscope as ps
import igl
import os

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.math.vector3 as V3

from math import *

class GraphicsComponent:
    #Private:
    boneIdx = []
    bonePosition = []
    boneQuaternion = []
    volumes = []
    def callbackTest():
        print("HELLO")

    def shiftMesh(vertices, offsetLeftSurface, offsetRightSurface):
        """
        DEPRECATED
        """
        itemRange = (vertices.shape[0])/2
        if (itemRange % 2 != 0):
            raise ValueError("Shape is not correctly designed")
        for i in range(vertices.shape[0]):
            if (i < itemRange):
                vertices[i][0] += offsetLeftSurface[0]
                vertices[i][1] += offsetLeftSurface[1]
                vertices[i][2] += offsetLeftSurface[2]
            else:
                vertices[i][0] += offsetRightSurface[0]
                vertices[i][1] += offsetRightSurface[1]
                vertices[i][2] += offsetRightSurface[2]
        return vertices
        
    def positionMesh(self, vertices, offset):
        """
        Give a mesh (vertices), move the object with an offset.

        :param vertices:       The vertices of a bone mesh.
        :param offset:         The position a mesh should be placed at.
        
        :return:               The new position of the vertices.
        """
        for i in range(vertices.shape[0]):
            vertices[i][0] += offset[0]
            vertices[i][1] += offset[1]
            vertices[i][2] += offset[2]
        return vertices

    def rotateMesh(self, vertices, eulerAngles):
        """
        DEPRECATED
        Given a euler angles, rotate all vertices using the ZYZ rotation matrix.

        :param vertices:       The vertices of a bone mesh.
        :param eulerAngles:    The ZYZ rotation, stored as a 3D-vector containing alpha beta and gamma.
        
        :return:               The rotated mesh vertices.
        """
        alpha = eulerAngles[0]
        beta = eulerAngles[1]
        gamma = eulerAngles[2]
        c1 = cos(alpha)
        c2 = cos(beta)
        c3 = cos(gamma)
        s1 = sin(alpha)
        s2 = sin(beta)
        s3 = sin(gamma)
        #Assuming only ZYZ Euler angles are being used  . 
        ZYZ_mat = np.matrix([
            [c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
            [c1*s3 + c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
            [-c3*s2, s2*s3, c2]
        ])
        for i in range(vertices.shape[0]):
            vertices[i] = vertices[i] * ZYZ_mat
        return vertices
        
    def rotateMeshQ(self, vertices, quaternion):
        """
        Given a quaternion, rotate all vertices applying a rotation matrix 
        quaternion.

        :param vertices:       The vertices of a bone mesh.
        :param quaternion:     The quaternion, stored as a 4D-vector in the form WXYZ.
        
        :return:               The rotated mesh vertices.
        """
        #Assuming Q_wcs is on the form WXYZ
        qr = quaternion[0]
        qi = quaternion[1]
        qj = quaternion[2]
        qk = quaternion[3]
        quaternion = 1/np.linalg.norm(quaternion) 
#        quaternion = np.sqrt(1.0/np.dot(quaternion, quaternion))
        #s = pow(np.linalg.norm(quaternion), -2)
        #s = 1.0/np.dot(quaternion, quaternion)
        s = 1.0
#        s = 1.0
#        s = 
        
        """https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix"""
        R_mat = np.matrix([
            [1.0 - 2.0*s*(qj*qj + qk*qk), 2.0*s*(qi*qj - qk*qr), 2.0*s*(qi*qk + qj*qr)],
            [2.0*s*(qi*qj + qk*qr), 1.0 - 2.0*s*(qi*qi + qk*qk), 2.0*s*(qj*qk - qi*qr)],
            [2.0*s*(qi*qk - qj*qr), 2.0*s*(qj*qk + qi*qr), 1.0 - 2.0*s*(qi*qi + qj*qj)]
        ])
        
        for i in range(vertices.shape[0]):
            vertices[i] = vertices[i] * -1 * np.transpose(R_mat)
        return vertices
        
    def createBoneMesh(self, pos, boneLength, quaternion):
        """
        Create the vertices for a bone mesh at the desired position. 

        :param pos:            A 3D-vector position stored as a numpy array.
        :param quaternion:     The quaternion representing the rotation of the bone.
        :param boneLength:     The length of a bone.

        :return:               The position of the vertices and the corresponding cells.
        """
        dist = boneLength[0]
        extend = 0.075
        vertices = np.array([
            [0, -extend, -extend],
            [0, -extend, extend],
            [0, extend, extend],
            [0, extend, -extend],
            [-dist, -extend, -extend],
            [-dist, -extend, extend],
            [-dist, extend, extend],
            [-dist, extend, -extend]
        ])
        print(pos)

#        startPos = self.rotateMeshQ(np.array([pos]), quaternion)[0]
#        normal = np.array([startPos[0]+1, 0, 0])-startPos
#        normal = normal/np.linalg.norm(normal)
#        normal = np.linalg.norm(np.cross(startPos, np.array([startPos[0], startPos[1], startPos[2] + 0.15])))
#        print(normal)
#        endPos = startPos + normal*1
#        vertices = np.array([
#            [startPos[0], startPos[1], startPos[2]],
#            [startPos[0], startPos[1], startPos[2] + 0.15],
#            [startPos[0], startPos[1] + 0.15, startPos[2] + 0.15],
#            [startPos[0], startPos[1] + 0.15, startPos[2]],
#            [endPos[0], endPos[1], endPos[2]],
#            [endPos[0], endPos[1], endPos[2] + 0.15],
#            [endPos[0], endPos[1] + 0.15, endPos[2] + 0.15],
#            [endPos[0], endPos[1] + 0.15, endPos[2]]
#        ])
        
        # Position mesh
        vertices = self.rotateMeshQ(vertices, quaternion)
        vertices = self.positionMesh(vertices, pos)
        cells = np.array([
          [0, 1, 2, 3, 4, 5, 6, 7],
        ])

        return (vertices, cells)

    def createBone(self, pos, nextBonePos, eulerAngles):
        """
        DEPRECATED
        Create the vertices for a bone mesh at the desired position. 

        :param pos:            A 3D-vector position stored as a numpy array.
        :param eulerAngles:    Rotation in the ZYZ-angles

        :return:               The position of the vertices and the corresponding cells.
        """
        dist = np.linalg.norm(pos-nextBonePos)
        vertices = np.array([
            [0, 0, 0],
            [0.15, 0, 0],
            [0.15, 0.15, 0],
            [0, 0.15, 0],
            [0, 0, dist],
            [0.15, 0, dist],
            [0.15, 0.15, dist],
            [0, 0.15, dist]
        ])
        
        # Add offset
        vertices = positionMesh(vertices, pos)
        vertices = rotateMesh(vertices, eulerAngles)
        cells = np.array([
          [0, 1, 2, 3, 4, 5, 6, 7],
        ])

        return (vertices, cells)
    
    def constructBoneMeshes(self):
        bones_amnt = len(self.boneIdx)
        for i in range(bones_amnt):
#            if (i == 0):
                #First bone, soo it does not need to be positioned based on the previous bone:
#                verts, cells = self.createBoneMesh(self.bonePosition[i], self.boneQuaternion[i])
#                self.volumes.append(ps.register_volume_mesh(("Bone_" + str(self.boneIdx[i])), verts, mixed_cells=cells, enabled=True, 
#                                 color=(0.3, 0.6, 0.8), interior_color=(0.4, 0.7, 0.9),
#                                 edge_color=((0.8, 0.8, 0.8)), edge_width=1.0, 
#                                 material='wax', transparency=0.5))
            if (i != bones_amnt):
                verts, cells = self.createBoneMesh(self.bonePosition[i], self.boneLen[i], self.boneQuaternion[i])
#                verts, cells = self.createBoneMesh(self.rotateMeshQ(np.array([self.bonePosition[i]]), self.boneQuaternion[i])[0], self.boneLen[i], self.boneQuaternion[i])
#                verts, cells = self.createBoneMesh(self.rotateMeshQ(np.array([self.bonePosition[i]]), self.boneQuaternion[i])[0], self.boneQuaternion[i])
                self.volumes.append(ps.register_volume_mesh(("Bone_" + str(self.boneIdx[i])), verts, mixed_cells=cells, enabled=True, 
                                 color=(0.3, 0.6, 0.8), interior_color=(0.4, 0.7, 0.9),
                                 edge_color=((0.8, 0.8, 0.8)), edge_width=1.0, 
                                 material='wax', transparency=0.5))
    
    def generateSkeletonMesh(self, skeleton):
        """
        Given a skeleton, use store the position and oritation of the contained bones

        :param skeleton:       A skeleton class defined consisting of bones.

        :return:               Void.
        """
#        if not (skeleton.has_bone()):
#            raise ValueError("Error when creating a skeleton mesh - a minimum of one bone is required.")
        #If this is called again, simply delete the old data:
        self.bonePosition = []
        self.boneQuaternion = []
        self.boneLen = []
        self.boneIdx = []
        
        for i in skeleton.bones:
            self.bonePosition.append(i.t_wcs)
            self.boneQuaternion.append(i.q_wcs)
            self.boneLen.append(i.t)
            self.boneIdx.append(i.idx)
            print("Q_WCS[i]: " + str(i.t_wcs))
        self.constructBoneMeshes()

    def visualizeDeprecated(self):
        """
        Visualize the constructed vertices.

        :return:               Void.
        """
        ps.init()
        
    #    print(os.getcwd())
    #    verts, _, n, tets, _, _ = igl.read_obj(os.getcwd() + "/bones.obj")
    #    ps_vol = ps.register_volume_mesh("A", verts, tets=tets)
        
        verts, cells = createBone(np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([0, 0, 0]))
        print(verts)
        ps.register_volume_mesh("Bone_001", verts, mixed_cells=cells, enabled=True, 
                                 color=(0.3, 0.6, 0.8), interior_color=(0.4, 0.7, 0.9),
                                 edge_color=((0.8, 0.8, 0.8)), edge_width=1.0, 
                                 material='wax', transparency=0.5)

        verts, cells = createBone(np.array([1, 1, 1]), np.array([2, 2, 2]), np.array([0, 0.15, 1]))
        ps.register_volume_mesh("Bone_002", verts, mixed_cells=cells, enabled=True, 
                                 color=(1., 0., 0.), interior_color=(0., 1., 0.),
                                 edge_color=((0.8, 0.8, 0.8)), edge_width=1.0, 
                                 material='candy', transparency=0.5)

        verts, cells = createBone(np.array([0, 0.15, 2]), np.array([3, 3, 3]), np.array([0, 0, 2]))
        ps.register_volume_mesh("Bone_003", verts, mixed_cells=cells, enabled=True, 
                                 color=(0.3, 0.6, 0.8), interior_color=(0.4, 0.7, 0.9),
                                 edge_color=((0.8, 0.8, 0.8)), edge_width=1.0, 
                                 material='wax', transparency=0.5)
        #ps.set_user_callback(printTest)

        ps.show()

        ps.clear_user_callback()
    
    def createSkeletonJoints(self):
        ps.register_point_cloud("Debug_Joints", np.array(self.bonePosition), 
                                radius=0.035, enabled=True, 
                                 color=(0.3, 0.6, 0.3), 
                                 material='wax', transparency=0.8)
        
    def visualize(self):
        """
        Visualize the constructed vertices.

        :return:               Void.
        """
        ps.init()
        
        self.createSkeletonJoints()
#        print(os.path.dirname(os.getcwd()) + "/data/buddha.obj")
#        verts, _, n, tets, _, _ = igl.read_obj(os.path.dirname(os.getcwd()) + "/data/buddha.obj")
#        ps_vol = ps.register_volume_mesh("A", verts, tets=tets)
        
    #    print(os.getcwd())
    #    verts, _, n, tets, _, _ = igl.read_obj(os.getcwd() + "/bones.obj")
    #    ps_vol = ps.register_volume_mesh("A", verts, tets=tets)
        
        ps.show()

        ps.clear_user_callback()
