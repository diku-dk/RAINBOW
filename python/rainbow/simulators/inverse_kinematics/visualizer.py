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
    
    def __init__(self):
#        verts, cells = self.createBoneMesh(np.array([0.0, 0.0, 0.0]), np.array([2, 0, 0]))
        verts, faces = self.generate_cone_vertices(1, 2, 0.5, 1)
        ps.register_surface_mesh(("dummy"), verts, faces, enabled=True, color=(1.0, 1.0, 1.0), edge_color=((0.3, 0.8, 0.3)), smooth_shade=True, edge_width=0.0, material='ceramic')
                                 
    def updatePointCloud(self):
        """
        The user have called an update to the IK-skeleton. Update the points 
        """
##        self.m_pointCloud.update_point_positions(np.array(self.bonePosition[1:]))
        self.m_pointCloud.update_point_positions(np.array(self.bonePosition))
        verts, cells = self.createBoneMesh(np.array([0, 0, 0]), self.bonePosition[0])
        ps.register_surface_mesh(("SECONDBONE"), verts, cells, enabled=True, 
                  color=(1.0, 1.0, 1.0), edge_color=((0.3, 0.8, 0.3)), 
                  smooth_shade=True, edge_width=0.0, material='ceramic')

        
    
    def callback(self):
        """
        A callback-function which is called once every frame. Used to handle logic
        when something is shown on screen.
        """
#        IK.update_skeleton(self.m_skeleton)
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
                bone.alpha = boneRot[i][0]
                bone.beta = boneRot[i][1]
                bone.gamma = boneRot[i][2]
            IK.update_skeleton(self.m_skeleton)
            print(len(self.m_skeleton.bones))
            IK.solve(self.m_chains, self.m_skeleton)
            self.generateSkeletonMesh(self.m_skeleton, self.m_chains)
            self.updatePointCloud()

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

    def generate_cone_vertices_single_point(self, radius, height, segments):
        """
        DEPRECATED: Based on a radius and a single point.

        :param radius:         The bottom radius of the cone.
        :param height:         The height of the cone.
        :param segments:       The amount of faces in a cone.
        
        :return:               The vertices and indices of vertices forming 
                               triangles, forming a cone.
        """
        vertices = [np.array([-height, 0.0, 0.0])]
        tris = []
        steps = 2 * math.pi / segments
        for i in range(segments):
            step = i * steps
            vert = np.array([0, radius * math.cos(step), radius * math.sin(step)])
            vert = np.array([height, radius * math.cos(step), radius * math.sin(step)])
            vertices.append(vert)
            if i != segments-1:
                tris.append(np.array([i, 0, i+1]))
            else:
                tris.append(np.array([i, 0, 1]))
#                tris.append(np.array([i, 0, i]))

    def generate_cone_vertices(self, radius, topRadius, height, segments):
        """
        Based on the radius of the bottom and top of the cylinder-cone.

        :param radius:         The radius of the bottom of the cone.
        :param topRadius:      The radius of the top cone.
        :param height:         The height of the cone.
        :param segments:       The amount of faces in a cone.
        
        :return:               The vertices and indices of vertices forming 
                               triangles, forming a cone.
        """
        vertices = [np.array([height, 0.0, 0.0])]
        tris = []
        steps = 2 * math.pi / segments
        for i in range(segments):
            step = i * steps
            vert = np.array([0, radius * math.cos(step), radius * math.sin(step)])
            vertices.append(vert)
            
        steps = 2 * math.pi / segments
        for i in range(segments):
            step = i * steps
            vert = np.array([height, topRadius * math.cos(step), topRadius * math.sin(step)])
            vertices.append(vert)
        
        for i in range(segments):
            if i != segments-1:
                tris.append(np.array([i+1, i, i+segments]))
                tris.append(np.array([i+1, i+segments, i+segments+1]))
            else:
                tris.append(np.array([i, i+segments, 1]))
                tris.append(np.array([i+segments, segments+1, 1]))
#                tris.append(np.array([i, 0, i]))

        return np.array(vertices), np.array(tris)

        print(tris)
                
        return np.array(vertices), np.array(tris)
        
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

        cells = np.array([
          [0, 1, 2, 3, 4, 5, 6, 7],
        ])
        
        return self.generate_cone_vertices(0.25, 0.075, dist, 32)

        #return (vertices, cells)
    
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
        rotMat = self.convertQuaternionToRotMat(quaternion)
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
                vol = ps.register_surface_mesh(("Bone_" + str(self.boneIdx[i])), verts, cells, enabled=True, 
                          color=(1.0, 1.0, 1.0), edge_color=((0.3, 0.8, 0.3)), 
                          smooth_shade=True, edge_width=0.0, material='ceramic')
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
        ps.register_point_cloud("Test", 
                                             np.array(np.array([self.m_chains[0].goal])), 
                                             radius=0.35, enabled=True, 
                                             color=(0.3, 0.6, 0.3),
                                             material='ceramic', transparency=0.9)
        """
        Create bone going from 0, 0, 0 in WCS to the root bone:
        """
        verts, cells = self.createBoneMesh(np.array([0, 0, 0]), self.bonePosition[0])
        ps.register_surface_mesh(("FIRSTBONE"), verts, cells, enabled=True, 
                  color=(1.0, 1.0, 1.0), edge_color=((0.3, 0.8, 0.3)), 
                  smooth_shade=True, edge_width=0.0, material='ceramic')
        
        
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

    m_oldVolumeTransform = []
    m_volumeTransform = []
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
    
    def isTransformEqual(self, v1, v2):
        """
        Compare two transform matrices to see if they are almost equal.

        :param v1:             First transform
        :param v2:             Second transform      
        
        :return:               A boolean with 1 indicating the two transforms are 
                               the same and 0 means they are not the same.
        """
        v1Len = v1.shape
        if v1Len != v2.shape:
            raise ValueError("Dimensions of transforms does not align.")
        boolAcc = 0
        for i in range(v1.shape[0]):
            for j in range(v1.shape[1]):
                boolAcc += self.compareFloats(v1[i, j], v2[i, j])
        if (boolAcc != 0):
            return 0
        return 1
    
    def checkBoneTransform(self, v1, v2):
        """
        Check if the bone parameters have changed since last time -- This can 
        happen if either the bone is moved from the UI Panel or if the gizmo 
        transform the bone.
        
        :param v1:             The mat4x4 of rotation/position.
        :param v2:             The mat4x4 of rotation/position.

        :return:               0/false if there has not been a change, 1/true if
                               there has been a change.  
        """
        if (not self.isTransformEqual(v1, v2)):
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
        
    def extractRotationFromTransform(self, rotMat):
        """
        Convert a 4x4 translation matrix to ZYZ euler angles.
        
        :param rotMat:         The transformation rotation matrix.

        :return:               ZYZ-angles as a three seperate rotations.
        """
        rotMat = rotMat*-1
        transform = np.matrix([[rotMat[0, 0], rotMat[0, 1], rotMat[0, 2]],
                               [rotMat[1, 0], rotMat[1, 1], rotMat[1, 2]],
                               [rotMat[2, 0], rotMat[2, 1], rotMat[2, 2]]])
        thetaY = 0.0
        thetaZ0 = 0.0
        thetaZ1 = 0.0
        if (transform[2, 2] < 1.0):
            if (transform[2, 2] > -1.0):
                thetaY = math.acos(transform[2, 2])
                thetaZ0 = math.atan2(transform[1, 2], transform[0, 2])
                thetaZ1 = math.atan2(transform[2, 1], -transform[2, 0])
            else:
                thetaY = math.pi
                thetaZ0 = math.atan2(transform[1, 0], transform[1, 1])
                thetaZ1 = 0.0
        else:
            thetaY = 0.0
            thetaZ0 = math.atan2(transform[1, 0], transform[1, 1])
            thetaZ1 = 0.0
        
        return (thetaZ0, thetaY, thetaZ1)
        
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


        if(psim.Button("Solve IK")):
            # This is reached once when the button is pressed
            print("Solving IK...")
            # Annoying hack to allow for the gizmo to function properly
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
                volume = ps.get_surface_mesh("Bone_" + str(self.m_ikBoneName[i]))
                self.m_volumeTransform.append(volume.get_transform())
                self.m_volumePos.append(volume.get_position())
                
                psim.TreePop()
                psim.PopItemWidth()
        psim.PopItemWidth()
        
        #Kind of a hack, but is needed to allow for smoothly moving the bones with a gizmo:
        hasTransformed = 0
        if (self.m_volumeTransform != [] and self.m_oldVolumeTransform != []):
            for i in range(len(self.m_volumeTransform)):
                hasTransformed += self.checkBoneTransform(self.m_oldVolumeTransform[i], self.m_volumeTransform[i])

                if (hasTransformed != 0):
                    self.m_ignoreTransformBone = self.m_ikBoneName[i]
                    self.m_ikBonePos[i] = [self.m_volumeTransform[i][0, 3], self.m_volumeTransform[i][1, 3], self.m_volumeTransform[i][2, 3]]
                    """https://www.geometrictools.com/Documentation/EulerAngles.pdf"""
                    ZYZ = self.extractRotationFromTransform(ps.get_surface_mesh("Bone_" + str(self.m_ikBoneName[i])).get_transform())

                    self.m_ikBoneRotationDeg[i] = [IK.radians_to_degrees(ZYZ[0]), IK.radians_to_degrees(ZYZ[1]), IK.radians_to_degrees(ZYZ[2])]
                    self.m_ikBoneRotation[i] = [ZYZ[0], ZYZ[1], ZYZ[2]].copy()
                    
        
        self.m_oldVolumeTransform = self.m_volumeTransform.copy()
        self.m_volumeTransform = []
        self.m_volumePos = []
        
        if (hasTransformed != 0):
            return True
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
        
