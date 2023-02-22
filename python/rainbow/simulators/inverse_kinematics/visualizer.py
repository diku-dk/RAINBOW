import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import igl
import os
import math

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.math.vector3 as V3


from math import *

class GraphicsComponent:
    #Private:
    boneIdx = []
    bonePosition = []
    boneQuaternion = []
    boneEulerRot = []
    volumes = []
    m_callback = None
    m_skeleton = None
    m_chains = None
    m_pointCloud = None
    
    m_ignoreBoneIdx = None

    def updateMeshes(self):
        """
        The user have called an update to the IK-skeleton. Update the meshes 
        """
        self.m_pointCloud.update_point_positions(np.array(self.bonePosition))
        #A better method is probably needed. Right now every time the skeleton
        # is updated, the previous volumes are simply deleted and the meshes are
        # removed. It would be better to be able to move the mesh, but I have not
        # been able to find a method to rotate meshes:
#        volumes = []
        #print(len(self.volumes))
        #print(self.volumes)
        #print(len(self.bonePosition))
#        self.m_ignoreBoneIdx = self.m_callback.getIgnoreTransformBone()
#        for i in range(len(self.volumes)):
#            verts, cells = self.createBoneMesh(self.bonePosition[i], self.bonePosition[i+1], self.boneQuaternion[i])
            
            #CANT USE: VolumeMesh.update_vertex_positions(newPos)   
            #Temporary (permanent?) hack until I can figure out how to apply rotation to a mesh:
#            ps.remove_volume_mesh("Bone_" + str(self.boneIdx[i]))
#            vol = ps.register_volume_mesh(("Bone_" + str(self.boneIdx[i])), verts, mixed_cells=cells, enabled=True, 
#                             color=(0.3, 0.6, 0.8), interior_color=(0.4, 0.7, 0.9),
#                             edge_color=((0.3, 0.8, 0.3)), edge_width=1.0, 
#                             material='ceramic', transparency=0.8)
#            self.volumes.append(vol)
#            if self.m_ignoreBoneIdx is not None and self.m_ignoreBoneIdx != self.boneIdx[i]:
#                self.transformBoneMesh(self.volumes[i], self.bonePosition[i], self.bonePosition[i], self.boneQuaternion[i])
    
    def callback(self):
        """
        A callback-function which is called once every frame. Used to handle logic
        when something is shown on screen.
        """
        shouldUpdateSkeleton = self.m_callback.update()   
        if shouldUpdateSkeleton:
            self.m_ignoreBoneIdx = self.m_callback.getIgnoreTransformBone()
            if (len(self.boneIdx) != len(self.bonePosition) or len(self.boneIdx) != len(self.boneEulerRot)):
                raise ValueError("Bone information is misaligned.")
            boneIdxs, boneRot, bonePos = self.m_callback.getBonesInfo()
            for i in range(len(boneIdxs)):
                if not (self.m_skeleton.has_bone(self.boneIdx[i])):
                    raise ValueError("Bone does not exist in skeleton.")
                bone = self.m_skeleton.bones[self.boneIdx[i]]
                bone.idx = boneIdxs[i]
                bone.t = V3.make(bonePos[i][0], bonePos[i][1], bonePos[i][2])
                #To allow for smooth movement of gizmo, we need to retrieve the transform instead of the stored position for ik bones.'
                
                bone.alpha = boneRot[i][0]
                bone.beta = boneRot[i][1]
                bone.gamma = boneRot[i][2]
            IK.update_skeleton(self.m_skeleton)
            self.generateSkeletonMesh(self.m_skeleton, self.m_chains)
            self.updateMeshes()
            
            
            

    def positionMesh(self, vertices, offset):
        """
        Given a mesh (vertices), move the object with an offset.

        :param vertices:       The vertices of a bone mesh.
        :param offset:         The position a mesh should be placed at.
        
        :return:               The new position of the vertices.
        """
        for i in range(vertices.shape[0]):
            vertices[i][0] += offset[0]
            vertices[i][1] += offset[1]
            vertices[i][2] += offset[2]
        return vertices
        
    def rotateMeshQ(self, vertices, quaternion):
        """
        DEPRECATED: Given a quaternion, rotate all vertices applying a rotation matrix 
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
        s = 1.0
        
        """https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix"""
        R_mat = np.matrix([
            [1.0 - 2.0*s*(qj*qj + qk*qk), 2.0*s*(qi*qj - qk*qr), 2.0*s*(qi*qk + qj*qr)],
            [2.0*s*(qi*qj + qk*qr), 1.0 - 2.0*s*(qi*qi + qk*qk), 2.0*s*(qj*qk - qi*qr)],
            [2.0*s*(qi*qk - qj*qr), 2.0*s*(qj*qk + qi*qr), 1.0 - 2.0*s*(qi*qi + qj*qj)]
        ])
        
        for i in range(vertices.shape[0]):
            vertices[i] = vertices[i] * -1 * np.transpose(R_mat)
        return vertices

    def convertQuaternionToRotMat(self, quaternion):
        """
        DEPRECATED: Given a quaternion, create a rotation matrix from the quaternion.

        :param quaternion:     The quaternion, stored as a 4D-vector in the form WXYZ.
        
        :return:               The 3x3 rotation matrix.
        """
        #Assuming Q_wcs is on the form WXYZ
        qr = quaternion[0]
        qi = quaternion[1]
        qj = quaternion[2]
        qk = quaternion[3]
        quaternion = 1/np.linalg.norm(quaternion) 
        s = 1.0
        
        """https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix"""
        R_mat = np.matrix([
            [1.0 - 2.0*s*(qj*qj + qk*qk), 2.0*s*(qi*qj - qk*qr), 2.0*s*(qi*qk + qj*qr)],
            [2.0*s*(qi*qj + qk*qr), 1.0 - 2.0*s*(qi*qi + qk*qk), 2.0*s*(qj*qk - qi*qr)],
            [2.0*s*(qi*qk - qj*qr), 2.0*s*(qj*qk + qi*qr), 1.0 - 2.0*s*(qi*qi + qj*qj)]
        ])
        
        return R_mat
        
    def createBoneMesh(self, boneStartPos, boneEndPos):
        """
        Create the vertices for a bone mesh at the desired position. 

        :param quaternion:     The quaternion representing the rotation of the bone.
        :param boneStartPos:   The start position of a bone.
        :param boneEndPos:     The end position of a bone.

        :return:               The position of the vertices and the corresponding cells.
        """

        dist = math.dist(boneStartPos, boneEndPos) 
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

        # Position mesh
#        vertices = self.rotateMeshQ(vertices, quaternion)
#        vertices = self.positionMesh(vertices, pos)
        cells = np.array([
          [0, 1, 2, 3, 4, 5, 6, 7],
        ])

        return (vertices, cells)
    
    def transformBoneMesh(self, meshVolume, boneStartPos, boneEndPos, quaternion):
        """
        Move the vertices for a bone mesh to the desired position by creating a
        transform matrix. 

        :param meshVolume:     The polyscope representation of a bone.
        :param boneStartPos:   The start position of a bone.
        :param boneEndPos:     The end position of a bone.
        :param quaternion:     The quaternion representing the rotation of the bone.
        
        :return:               The updated volume.
        """
        rotMat = -1*self.convertQuaternionToRotMat(quaternion)
 #       rotMat = -1*np.transpose(rotMat)
        transform = np.matrix([[rotMat[0, 0], rotMat[0, 1], rotMat[0, 2], boneStartPos[0]],
                               [rotMat[1, 0], rotMat[1, 1], rotMat[1, 2], boneStartPos[1]],
                               [rotMat[2, 0], rotMat[2, 1], rotMat[2, 2], boneStartPos[2]],
                               [0.0, 0.0, 0.0, 1.0]])
        
        meshVolume.set_transform(transform)
    
    def constructBoneMeshes(self):
        """
        Given some bones, construct and register the actual mesh of the bones
        """
        bones_amnt = len(self.boneIdx)
        if (self.volumes == []):
            for i in range(bones_amnt-1):
                verts, cells = self.createBoneMesh(self.bonePosition[i], self.bonePosition[i+1])
                vol = ps.register_volume_mesh(("Bone_" + str(self.boneIdx[i])), verts, mixed_cells=cells, enabled=True, 
                                 color=(0.3, 0.6, 0.8), interior_color=(0.4, 0.7, 0.9),
                                 edge_color=((0.3, 0.8, 0.3)), edge_width=1.0, 
                                 material='ceramic', transparency=0.8)
                self.transformBoneMesh(vol, self.bonePosition[i], self.bonePosition[i+1], self.boneQuaternion[i])
                self.volumes.append(vol)
        else:
            for i in range(bones_amnt-1):
                if not (self.m_ignoreBoneIdx is not None and self.m_ignoreBoneIdx == self.boneIdx[i]):
                    self.transformBoneMesh(self.volumes[i], self.bonePosition[i], self.bonePosition[i+1], self.boneQuaternion[i])
    
    def generateSkeletonMesh(self, skeleton, chains):
        """
        Given a skeleton, store the position and oritation of the contained bones

        :param skeleton:       A skeleton class defined consisting of bones.
        """
        #If this function is called multiple times, it means deleting the old data:
        self.bonePosition = []
        self.boneQuaternion = []
        self.boneIdx = []
        self.boneEulerRot = []
        self.m_skeleton = skeleton
        self.m_chains = chains
        
        for i in skeleton.bones:
            self.bonePosition.append(i.t_wcs)
            self.boneQuaternion.append(i.q_wcs)
            self.boneIdx.append(i.idx)
            self.boneEulerRot.append(np.array([i.alpha, i.beta, i.gamma]))
            print("Q_WCS[i]: " + str(i.t_wcs))
        self.constructBoneMeshes()
    
    def createSkeletonJoints(self):
        """
        Creates ball-spheres for visual purposes only.
        """
        self.m_pointCloud = ps.register_point_cloud("Debug_Joints", 
                                             np.array(self.bonePosition), 
                                             radius=0.035, enabled=True, 
                                             color=(0.3, 0.6, 0.3),
                                             material='ceramic', transparency=0.9)
        
    def visualize(self):
        """
        Visualize the constructed vertices.
        """
        ps.init()
        
        #Set callback function to give the user functionality for manipulating skeletons
        self.m_callback = CallbackHandler()
        #TODO: Make better system for IK bones
        self.m_callback.initIKBones([self.boneIdx[0]], [self.boneEulerRot[0]], [self.bonePosition[0]])
        ps.set_user_callback(self.callback)
#        ps.set_user_callback(self.updaterUI)
        
        self.createSkeletonJoints()
#        print(os.path.dirname(os.getcwd()) + "/data/buddha.obj")
#        verts, _, n, tets, _, _ = igl.read_obj(os.path.dirname(os.getcwd()) + "/data/buddha.obj")
#        ps_vol = ps.register_volume_mesh("A", verts, tets=tets)
        
    #    print(os.getcwd())
    #    verts, _, n, tets, _, _ = igl.read_obj(os.getcwd() + "/bones.obj")
    #    ps_vol = ps.register_volume_mesh("A", verts, tets=tets)
        #ps.set_user_callback(printTest)
        
        ps.show()

        ps.clear_user_callback()

class CallbackHandler:
    #Private
    m_animComponent = None
    m_ikBoneNameRef = []
    m_ikBoneRotationDegRef = []
    m_ikBonePosRef = []
    m_ikBoneName = []
    m_ikBoneRotation = []
    m_ikBoneRotationDeg = []
    m_ikBonePos = []

    m_oldVolumePos = []
    m_volumePos = []
    
    m_ignoreTransformBone = None
    
    def __init__(self):
        self.m_animComponent = AnimationComponent()
    
    def initIKBones(self, boneName, boneRot, bonePos):
        """
        Only the bones with IK applied, needs to have their position
        updated/stored. The rest of the bones does not

        :param bonePos:        IK-bones, which are the bone where the parents 
                               have their position and rotation updated based on 
                               this bone.
        :param boneRot:        The rotation of an IK-bone
        :param boneName:       The name of an IK-bone 
        """
        self.m_ikBoneName = boneName
        self.m_ikBoneRotation = boneRot
        for i in self.m_ikBoneRotation:
            self.m_ikBoneRotationDeg.append(np.array([IK.radians_to_degrees(i[0]), IK.radians_to_degrees(i[1]), IK.radians_to_degrees(i[2])]))
        self.m_ikBonePos = bonePos
#        self.m_ikBoneNameRef = boneName
#        self.m_ikBoneRotationRef = [np.array([0.5, 0.5, 0.5])]
#        self.m_ikBonePosRef = [np.array([0.5, 0.5, 0.5])]
        self.m_ikBoneNameRef = boneName.copy()
        self.m_ikBoneRotationDegRef = (self.m_ikBoneRotationDeg).copy()
        self.m_ikBonePosRef = bonePos.copy()
    
    def updateRotationRad(self):
        for i in range(len(self.m_ikBoneRotationDeg)):
            self.m_ikBoneRotation[i][0] = IK.degrees_to_radians(self.m_ikBoneRotationDeg[i][0])
            self.m_ikBoneRotation[i][1] = IK.degrees_to_radians(self.m_ikBoneRotationDeg[i][1])
            self.m_ikBoneRotation[i][2] = IK.degrees_to_radians(self.m_ikBoneRotationDeg[i][2])
             
    def getBonesInfo(self):
        """
        Retrieve all info of all IK bones.

        :return:               Return the name, position and rotation of the bone 
        """
        self.updateRotationRad()
        return (self.m_ikBoneName, self.m_ikBoneRotation, self.m_ikBonePos)
    
    def resetBones(self):
        self.updateRotationRad()
        self.m_ikBoneName = (self.m_ikBoneNameRef).copy()
        self.m_ikBoneRotationDeg = (self.m_ikBoneRotationDegRef).copy()
        self.m_ikBonePos = (self.m_ikBonePosRef).copy()
    
        
    def compareFloats(self, a, b):
        """
        Compare two floats to see if they are almost the same or the same.

        :param a:              First float
        :param b:              Second float        
        
        :return:               A boolean with 0 indicating the two floats are 
                               similar and 1 means they are not similar.
        """
        return (abs(a-b) > 0.000001)
    
    def isVectorsEqual(self, v1, v2):
        """
        Compare two floats to see if they are almost the same or the same.

        :param v1:             First vector
        :param v2:             Second vector        
        
        :return:               A boolean with 1 indicating the two vectors are 
                               the same and 0 means they are not the same.
        """
        v1Len = v1.size
        if v1Len != v2.size:
            raise ValueError("Dimensions of arrays does not align.")
        boolAcc = 0
        for i in range(v1Len):
            boolAcc += self.compareFloats(v1[i], v2[i])
        if (boolAcc != 0):
            return 0
        return 1
    
    def checkBoneTransform(self, v1, v2):
        """
        Check if the bone parameters have changed since last time -- This can 
        happen if either the bone is moved from the UI Panel or if the gizmo 
        transform the bone.
        
        :param v1:             The vector_1 of rotation/position.
        :param v2:             The vector_2 of rotation/position.

        :return:               0/false if there has not been a change, 1/true if
                               there has been a change.  
        """
        if (not self.isVectorsEqual(v1, v2)):
            return True
        return False
    
    def getIgnoreTransformBone(self):
        """
        Unfortunately this function, while being ugly, is needed to ensure smooth transform
        with a gizmo. If this was not here, the user would be unable to move an object with
        the gizmo.
         
        :return:               One IK-bone to ignore, as the user can only move one bone
                               At a time with the gizmo. If there is no bones to move,
                               return None.
        """ 
        return self.m_ignoreTransformBone
        
        
    def update(self):
        """
        This function is used to display the IK options panel, where the user
        can apply multiple functionality to a skeleton.
         
        :return:               True if any changes has been made to the rotation
                               and/or position and the user wants to solve IK/
                               Reset skeleton. This is so the GraphicsComponent
                               can recalculate the skeleton with
                               inverse kinematics.  
        """ 
        
        #Init:
        psim.PushItemWidth(150)

        psim.TextUnformatted("Options for IK bone")
        psim.Separator()
        psim.PopItemWidth()
        # == Buttons


        if(psim.Button("Solve IK")):
            # This is reached once when the button is pressed
            print("Solving IK...")
            # Annoying hack
            self.m_ignoreTransformBone = None 
            return True

        # Put next element on same line.
        psim.SameLine() 
        
        if(psim.Button("Reset skeleton")):
            # This is reached once when the button is pressed
            print("Reset positions")
            self.resetBones()
            return True
        psim.PopItemWidth()

        for i in range(len(self.m_ikBoneName)):
            psim.SetNextItemOpen(True, psim.ImGuiCond_FirstUseEver)

            # The body is executed only when the sub-menu is open. Note the push/pop pair!
            if(psim.TreeNode("Bone_" + str(self.m_ikBoneName[i]))):
                psim.PushItemWidth(150)
                psim.TextUnformatted("Transform for IK Bones")
                edited, self.m_ikBonePos[i] = psim.InputFloat3("Bone position", self.m_ikBonePos[i])
                
                edited, self.m_ikBoneRotationDeg[i] = psim.InputFloat3("Bone rotation", self.m_ikBoneRotationDeg[i])
       
       
                #Check for any change in the bone structure.
#                print(ps.get_volume_mesh("Bone_" + str(self.m_ikBoneName[i])).get_position())
#                print("AND")
#                print(np.array(self.m_ikBonePos[i]))
                volumePos = ps.get_volume_mesh("Bone_" + str(self.m_ikBoneName[i])).get_position()
                self.m_volumePos.append(volumePos)
                
                psim.TreePop()
                psim.PopItemWidth()
        psim.PopItemWidth()
        
        
        
        #Kind of a hack, but is needed to allow for smoothly moving the bones with a gizmo:
        hasTransformed = 0
        if (self.m_volumePos != [] and self.m_oldVolumePos != []):
            for i in range(len(self.m_volumePos)):
                hasTransformed += self.checkBoneTransform(self.m_oldVolumePos[i], self.m_volumePos[i])

                if (hasTransformed != 0):
                    self.m_ignoreTransformBone = self.m_ikBoneName[i]
                    self.m_ikBonePos[i] = [volumePos[0], volumePos[1], volumePos[2]]
        
        self.m_oldVolumePos = self.m_volumePos.copy()
        self.m_volumePos = []
        
        if (hasTransformed != 0):
            return True
            print("HELLO")  
        else:
            self.m_ignoreTransformBone = None 
        
        return False


class Keyframe:
    #Private
    def __init__(self, pos, rot, frame, bone):
        m_position = pos
        m_rotation = rot
        m_frameNum = frame
        m_bone = bone
    
    def getPosition(self):
        return self.m_position
    
    def getRotation(self):
        return self.m_rotation
    
    def getFrameNum(self):
        return self.m_frameNum
    
    def getBone(self):
        return self.m_bone
    

    
class AnimationComponent:
    #Private
    #List of keyframes
    m_keyframes = []
    m_currentFrame = 0
    m_lastFrame = 0
    
    
    def addKeyframe(self, keyframe):
        self.m_keyframes.append(keyframe)
        if keyframe.getFrameNum() > m_lastFrame:
            self.m_lastFrame = keyframe.getFrameNum()
        
    def removeKeyframe(self):
        #Remove the latest keyframe
        if (len(self.m_keyframes != 0)):
            self.m_keyframes[len(self.m_keyframes)-1]
    
    def update(self, dt):
        self.m_currentFrame += 1
        
