import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import igl
import os
import math

import rainbow.simulators.inverse_kinematics.api as IK
import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q


from math import *

class GraphicsComponent:
    #Private:
    boneIdx = []
    bonePosition = []
    boneQuaternion = []
    boneEulerRot = []
    volumes = []
    toolPositions = []
    toolRotations = []

    
    m_callback = None
    m_skeleton = None
    m_chains = None
    m_pointCloud = None
    
    def __init__(self):
        m_chains = None
                                 
    def updatePointCloud(self):
        """
        The user have called an update to the IK-skeleton. Update the points 
        """
        self.m_pointCloud.update_point_positions(np.array(self.bonePosition))
        verts, cells = self.createBoneMesh(np.array([0, 0, 0]), self.bonePosition[0])
        ps.register_surface_mesh(("ZeroZeroBone"), verts, cells, enabled=True, 
                  color=(1.0, 1.0, 1.0), edge_color=((0.3, 0.8, 0.3)), 
                  smooth_shade=True, edge_width=0.0, material='ceramic')

    def callback_updateLimits(self):
        names, lims = self.m_callback.getBoneLimits()
        if (len(lims) != len(self.m_skeleton.bones)):
            raise ValueError("Limits misaligned with the bones in the skeleton")
        for i in range(len(self.m_skeleton.bones)):
            self.m_skeleton.bones[i].set_limit_alpha(lims[i][0][0], lims[i][1][0])
            self.m_skeleton.bones[i].set_limit_beta(lims[i][0][1], lims[i][1][1])
            self.m_skeleton.bones[i].set_limit_gamma(lims[i][0][2], lims[i][1][2])
        
    def callback(self):
        """
        A callback-function which is called once every frame. Used to handle logic
        when something is shown on screen.
        """
        from timeit import default_timer

        
        start = default_timer()
        shouldUpdateSkeleton, transformationType = self.m_callback.update()   
        if shouldUpdateSkeleton:
            if (len(self.boneIdx) != len(self.bonePosition) or len(self.boneIdx) != len(self.boneEulerRot)):
                raise ValueError("Bone information is misaligned.")
            self.callback_updateLimits()
            meshName, goalPos = self.m_callback.getGoalInfo()
            for i in range(len(meshName)):
                self.m_chains[i].goal = V3.make(goalPos[i][0], goalPos[i][1], goalPos[i][2])
                #Only transform the bone, if the user is not handling the goal
                #with the gizmo.
                if (transformationType == 1):
                    self.transformBoneMesh(ps.get_surface_mesh(meshName[i]), goalPos[i], [1, 0, 0, 0])
            IK.update_skeleton(self.m_skeleton)
            IK.solve(self.m_chains, self.m_skeleton)
            self.generateSkeletonMesh(self.m_skeleton, self.m_chains)
            self.updatePointCloud()
            duration = default_timer() - start
            print("Elapsed total time including visualizer... " + str(duration))

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
        Create a cone based on a radius and a single point.

        :param radius:         The bottom radius of the cone.
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
            if i != segments-1:
                tris.append(np.array([i, 0, i+1]))
            else:
                tris.append(np.array([i, 0, 1]))
        return np.array(vertices), np.array(tris)

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

        return np.array(vertices), np.array(tris)
        
    def eulerAnglesToRotMat(self, alpha, beta, gamma):
        c1 = cos(alpha)
        c2 = cos(beta)
        c3 = cos(gamma)
        s1 = sin(alpha)
        s2 = sin(beta)
        s3 = sin(gamma)
        mat = np.matrix([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
        return mat
        
    def cmpFlt(self, a):
        return (abs(a) > 0.000001)
        
    def createBoneMesh(self, boneStartPos, boneEndPos):
        """
        Create the vertices for a bone mesh at the desired position. 

        :param quaternion:     The quaternion representing the rotation of the bone.
        :param boneStartPos:   The start position of a bone.
        :param boneEndPos:     The end position of a bone.

        :return:               The position of the vertices and the corresponding cells.
        """

        dist = math.dist(boneStartPos, boneEndPos)
        
        forward = boneStartPos-boneEndPos
        normalized = np.linalg.norm(forward)
        if (forward[0] > 0.000000001):
            forward[0] = forward[0] / normalized
        if (forward[1] > 0.000000001):
            forward[1] = forward[1] / normalized
        if (forward[2] > 0.000000001):
            forward[2] = forward[2] / normalized
        
        verts = self.generate_cone_vertices(0.25, 0.075, dist, 32)
        
        up = None
        mat = None
        if self.cmpFlt(forward[0]) == True:
            up = np.array([0, 1, 0])
        else:
            up = np.array([0, 0, 1])
        right = np.cross(up, forward)
        if (right[0] > 0.000000001):
            right[0] = right[0] / np.linalg.norm(right)
        if (right[1] > 0.000000001):
            right[1] = right[1] / np.linalg.norm(right) 
        if (right[2] > 0.000000001):
            right[2] = right[2] / np.linalg.norm(right)
        #right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        #up = up / np.linalg.norm(up)
        if (up[0] > 0.000000001):
            up[0] = up[0] / np.linalg.norm(up)
        if (up[1] > 0.000000001):
            up[1] = up[1] / np.linalg.norm(up)
        if (up[2] > 0.000000001):
            up[2] = up[2] / np.linalg.norm(up)
        mat = np.matrix([[up[0], up[1], up[2]],
                         [right[0], right[1], right[2]],
                         [forward[0], forward[1], forward[2]]])
        if (forward[0] > 0.999):
            mat = self.eulerAnglesToRotMat(0.0, IK.degrees_to_radians(forward[0]*180.0), 0.0)
        else:         
            mat = self.eulerAnglesToRotMat(IK.degrees_to_radians(forward[1]*90.0), 0.0, 0.0)
        for i in range(len(verts[0])):
            verts[0][i] = verts[0][i]*mat
        return verts
    
    def transformBoneMesh(self, meshVolume, boneStartPos, quaternion):
        """
        Move the vertices for a bone mesh to the desired position by creating a
        transform matrix. 

        :param meshVolume:     The polyscope representation of a bone.
        :param boneStartPos:   The start position of a bone.
        :param quaternion:     The quaternion representing the rotation of the bone.
        
        :return:               The updated volume.
        """
        rotMat = self.convertQuaternionToRotMat(quaternion)
        transform = np.matrix([[rotMat[0, 0], rotMat[0, 1], rotMat[0, 2], boneStartPos[0]],
                               [rotMat[1, 0], rotMat[1, 1], rotMat[1, 2], boneStartPos[1]],
                               [rotMat[2, 0], rotMat[2, 1], rotMat[2, 2], boneStartPos[2]],
                               [0.0, 0.0, 0.0, 1.0]])
        meshVolume.set_transform(transform)
        
    def drawBonesBetweenChains(self, skeleton, chains):
        """
        Given some chains, if the distance is more than 0.0001, draw bones between chains.
        """
        for i in range(len(chains)-1):
            chain = chains[i]
            for j in range(len(chain.bones)-2):
                if (abs(chain.bones[j]-chain.bones[j+1]) > 1):
                    vec = skeleton.bones[chain.bones[j]].t_wcs-skeleton.bones[chain.bones[j+1]].t_wcs
                    if (abs(vec[0]) > 0.0001 or abs(vec[1]) > 0.0001 or abs(vec[2]) > 0.0001):
                        verts, cells = self.createBoneMesh(skeleton.bones[chain.bones[j]].t_wcs, skeleton.bones[chain.bones[j+1]].t_wcs)
                        vol = ps.register_surface_mesh(("ConstantBone" + str(chain.bones[j]) + "_" + str(i)), verts, cells, enabled=True, 
                                  color=(1.0, 1.0, 1.0), edge_color=((0.3, 0.8, 0.3)), 
                                  smooth_shade=True, edge_width=0.0, material='ceramic')
                        self.transformBoneMesh(vol, skeleton.bones[chain.bones[j]].t_wcs, Q.identity())
 
    def constructBoneMeshes(self):
        """
        Given some bones, construct and register the actual mesh of the bones
        """
        bones_amnt = len(self.boneIdx)
        if (self.volumes == []):
            for i in range(bones_amnt-1):
                #Check if there is any child bones, if not, the mesh should not be created.
                if (len(self.m_skeleton.bones[self.boneIdx[i]].children) != 0):
                    verts, cells = self.createBoneMesh(self.bonePosition[i], self.bonePosition[i+1])
                    vol = ps.register_surface_mesh(("Bone_" + str(self.boneIdx[i])), verts, cells, enabled=True, 
                              color=(1.0, 1.0, 1.0), edge_color=((0.3, 0.8, 0.3)), 
                              smooth_shade=True, edge_width=0.0, material='ceramic')
                    self.transformBoneMesh(vol, self.bonePosition[i], self.boneQuaternion[i])
                    self.volumes.append(vol)
                else:
                    self.volumes.append(None)
        else:
            for i in range(len(self.volumes)):
                if self.volumes[i] is not None:
#                if (len(self.m_skeleton.bones[self.boneIdx[i]].children) != 0):
                    self.transformBoneMesh(self.volumes[i], self.bonePosition[i], self.boneQuaternion[i])
    
    def generateToolVectors(self, skeleton, chains):
        for i in range(len(chains)):
            if (chains[i].tool[0] + chains[i].tool[1] + chains[i].tool[2] > 0.1):
                bone = skeleton.bones[chains[i].bones[-1]]
                q_alpha = bone.get_rotation_alpha()
                q_beta = bone.get_rotation_beta()
                q_gamma = bone.get_rotation_gamma()
                q_bone = Q.prod(q_alpha, Q.prod(q_beta, q_gamma))

                t_parent = V3.zero()
                q_parent = Q.identity()
                t_parent = bone.t_wcs
                q_parent = bone.q_wcs
                t_wcs = t_parent + Q.rotate(q_parent, chains[i].tool)
                q_wcs = Q.prod(q_parent, q_bone)
                
                verts, cells = self.createBoneMesh(bone.t_wcs, t_wcs)
                vol = ps.register_surface_mesh(("Tool" + str(i)), verts, cells, enabled=True, 
                          color=(0.2, 1.0, 0.5), edge_color=((0.3, 0.8, 0.3)), 
                          smooth_shade=True, edge_width=0.0, material='ceramic')              
            
                self.transformBoneMesh(vol, bone.t_wcs, q_wcs)
        
    def generateSkeletonMesh(self, skeleton, chains):
        """
        Given a skeleton, store the position and oritation of the contained bones
        Requires to be invocated before solving.

        :param skeleton:       A skeleton class defined consisting of bones.
        """
        #If this function is called multiple times, it means deleting the old data:
        self.bonePosition = []
        self.boneQuaternion = []
        self.boneIdx = []
        self.boneEulerRot = []
        self.m_skeleton = skeleton
        self.m_chains = chains
        self.drawBonesBetweenChains(skeleton, chains)
        
        for i in skeleton.bones:
            self.bonePosition.append(i.t_wcs)
            self.boneQuaternion.append(i.q_wcs)
            self.boneIdx.append(i.idx)
            self.boneEulerRot.append(np.array([i.alpha, i.beta, i.gamma]))

        self.generateToolVectors(skeleton, chains)
        self.constructBoneMeshes()
        
        
    def update_mesh(self, skeleton, chains):
        """
        Given a skeleton, store the position and oritation of the contained bones
        Requires to be invocated after solving.

        :param skeleton:       A skeleton class defined consisting of bones.
        """
        self.generateSkeletonMesh(skeleton, chains)
    
    def createSkeletonJoints(self):
        """
        Creates ball-spheres for visual purposes only.
        """
        self.m_pointCloud = ps.register_point_cloud("Debug_Joints", 
                                             np.array(self.bonePosition), 
                                             radius=0.035, enabled=True, 
                                             color=(0.3, 0.6, 0.3),
                                             material='ceramic', transparency=0.9)
        """
        Create bone going from 0, 0, 0 in WCS to the root bone:
        """
        verts, cells = self.createBoneMesh(np.array([0, 0, 0]), self.bonePosition[0])
        ps.register_surface_mesh(("_First_Bone"), verts, cells, enabled=True, 
                  color=(1.0, 1.0, 1.0), edge_color=((0.3, 0.8, 0.3)), 
                  smooth_shade=True, edge_width=0.0, material='ceramic')
        
    def initGoals(self):
        """
        Initialize the goal meshes
        """
        goalNames = []
        goalPositions = []
        for i in range(len(self.m_chains)):
            goalNames.append("Goal_" + str(i))
            goalPositions.append(self.m_chains[i].goal)
            verts, faces = self.generate_cone_vertices_single_point(0.25, 2, 32)
            vol = ps.register_surface_mesh((goalNames[i]), verts, faces, enabled=True, color=(0.1, 1.0, 0.1), edge_color=((0.3, 0.8, 0.3)), smooth_shade=True, edge_width=0.1, material='candy')
            self.transformBoneMesh(vol, goalPositions[i], [1.0, 0.0, 0.0, 0.0])
        self.m_callback.initIKBones(goalNames, goalPositions)
    
    def shiftArrToDegrees(self, rad):
        for i in range(len(rad)):
            rad[i] = rad[i] * (180/np.pi)
        return rad
        
    def initLimits(self, skeleton):
        """
        Initalize limits for each bone
        """
        names = []
        limits = []
        for i in range(len(skeleton.bones)):
            names.append("Bone_" + str(i))
            lower = (skeleton.bones[i].get_limit_lower()).copy()
            upper = (skeleton.bones[i].get_limit_upper()).copy()
            lower = self.shiftArrToDegrees(lower)
            upper = self.shiftArrToDegrees(upper)
            limits.append([lower, upper])
        self.m_callback.initLimitBones(names, limits)
            
    def visualize(self):
        """
        Visualize the constructed vertices.
        """
        ps.init()
        
        #Set callback function to give the user functionality for manipulating skeletons
        self.m_callback = CallbackHandler()

        ps.set_user_callback(self.callback)
        self.initGoals()
        self.initLimits(self.m_skeleton)

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
    m_meshNameRef = []
    m_ikBonePosRef = []
    m_meshName = []
    m_ikBoneRotation = []
    m_ikBonePos = []

    m_oldVolumeTransform = []
    m_volumeTransform = []
    m_volumePos = []
    
    #Incoperate limits...
    m_ordinaryBoneNamesRef = []
    m_ordinaryBoneLimitsRef = []
    
    m_ordinaryBoneNames = []
    m_ordinaryBoneLimits = []
    
    def initIKBones(self, boneName, bonePos):
        """
        Only the bones with IK applied, needs to have their position
        updated/stored. The rest of the bones does not

        :param bonePos:        IK-bones, which are the bone where the parents 
                               have their position and rotation updated based on 
                               this bone.
        :param boneRot:        The rotation of an IK-bone
        :param boneName:       The name of an IK-bone 
        """
        self.m_meshName = boneName
        self.m_ikBonePos = bonePos

        self.m_meshNameRef = boneName.copy()
        self.m_ikBonePosRef = bonePos.copy()
        
    def initLimitBones(self, boneNames, boneLimits):
        """
        Store the limits of each bone from a skeleton in an array.

        :param boneNames:      The name of a bone 
        :param boneLimits:     The limiting rotation of a bone stored as 
                               [[alpha_min, beta_min, gamma_min], [alpha_max, 
                               beta_max, gamma_max]] in an array
        """
        self.m_ordinaryBoneNames = boneNames
        self.m_ordinaryBoneLimits = boneLimits 
        
        self.m_ordinaryBoneNamesRef = boneNames.copy()
#        self.m_ordinaryBoneLimitsRef = boneLimits.copy()
        #The only way to get a reference array.
        for i in range(len(self.m_ordinaryBoneLimits)):
            self.m_ordinaryBoneLimitsRef.append([(boneLimits[i][0]).copy(), (boneLimits[i][1]).copy()])
        
    def getGoalInfo(self):
        """
        Retrieve all info of all IK bones.

        :return:               Return the name, position and rotation of the bone 
        """
        return (self.m_meshName, self.m_ikBonePos)
        
    def getBoneLimits(self):
        """
        Retrieve the updated limits of all IK bones.
        
        :return:               The names and limits. 
        """
        return (self.m_ordinaryBoneNames, self.m_ordinaryBoneLimits)
    
    def resetBones(self):
        #Reset bone orientation
        self.m_meshName = (self.m_meshNameRef).copy()
        self.m_ikBonePos = (self.m_ikBonePosRef).copy()
        
        #Reset limits as well
#        self.m_ordinaryBoneNames = (self.m_ordinaryBoneNamesRef).copy()
#        self.m_ordinaryBoneLimits = (self.m_ordinaryBoneLimitsRef).copy()
        for i in range(len(self.m_ordinaryBoneLimits)):
            self.m_ordinaryBoneLimits[i][0] = (self.m_ordinaryBoneLimitsRef[i][0]).copy()
            self.m_ordinaryBoneLimits[i][1] = (self.m_ordinaryBoneLimitsRef[i][1]).copy()
    
        
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
                               Reset skeleton. Furthermore it returns an int 
                               depending on the type of transformatio happening.
                               This is so the GraphicsComponent can recalculate 
                               the skeleton with inverse kinematics.
                               0 = Should not transform
                               1 = Should transform
        """ 
        
        #Init:
        psim.PushItemWidth(150)

        psim.TextUnformatted("Options for IK bone")
        psim.Separator()
        psim.PopItemWidth()


        if(psim.Button("Solve IK")):
            # This is reached once when the button is pressed
            print("Solving IK...") 
            return (True, 1)

        # Put next element on same line.
        psim.SameLine() 
        
        if(psim.Button("Reset skeleton")):
            # This is reached once when the button is pressed
            print("Reset positions")
            self.resetBones()
            return (True, 1)
        psim.PopItemWidth()

        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
        if(psim.TreeNode("Goal positions")):
            for i in range(len(self.m_meshName)):
                psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)

                # The body is executed only when the sub-menu is open. Note the push/pop pair!
                if(psim.TreeNode(self.m_meshName[i])):
                    psim.PushItemWidth(150)
                    psim.TextUnformatted("Transform for IK Goal Bone " + str(self.m_meshName[i]))
                    edited, self.m_ikBonePos[i] = psim.InputFloat3("Bone position", self.m_ikBonePos[i])
                    
                    psim.PopItemWidth()
                    psim.TreePop()
            psim.TreePop()
        
        #Kind of a hack, but is needed to allow for smoothly moving the bones with a gizmo:
        for i in range(len(self.m_meshName)):
            #Check for any change in the bone structure.
            volume = ps.get_surface_mesh(self.m_meshName[i])
            self.m_volumeTransform.append(volume.get_transform())
            self.m_volumePos.append(volume.get_position())

        hasTransformed = 0
        if (self.m_volumeTransform != [] and self.m_oldVolumeTransform != []):
            for i in range(len(self.m_volumeTransform)):
                hasTransformed += self.checkBoneTransform(self.m_oldVolumeTransform[i], self.m_volumeTransform[i])
                if (hasTransformed != 0):
                    self.m_ikBonePos[i] = [self.m_volumeTransform[i][0, 3], self.m_volumeTransform[i][1, 3], self.m_volumeTransform[i][2, 3]]
        
        self.m_oldVolumeTransform = self.m_volumeTransform.copy()
        self.m_volumeTransform = []
        self.m_volumePos = []

        #Apply limits
        psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)
        if(psim.TreeNode("Bone limits")):
            for i in range(len(self.m_ordinaryBoneNames)):
                psim.SetNextItemOpen(False, psim.ImGuiCond_FirstUseEver)

                # The following is only executed when the sub-menu is open.
                if(psim.TreeNode(self.m_ordinaryBoneNames[i])):
                    psim.PushItemWidth(150)
                    psim.TextUnformatted("Limit for bones")
                    edited, self.m_ordinaryBoneLimits[i][0] = psim.InputFloat3("Minimum bone limit", self.m_ordinaryBoneLimits[i][0])
                    edited, self.m_ordinaryBoneLimits[i][1] = psim.InputFloat3("Maximum bone limit", self.m_ordinaryBoneLimits[i][1])
                    
                    #Check for any change in the bone structure.
    #                volume = ps.get_surface_mesh(self.m_ordinaryBoneNames[i])
    #                self.m_volumeTransform.append(volume.get_transform())
    #                self.m_volumePos.append(volume.get_position())
                    
                    psim.PopItemWidth()
                    psim.TreePop()
            if(psim.Button("Apply Limits")):
                # This is reached once when the button is pressed 
                return (True, 1)
            psim.TreePop()
        
        if (hasTransformed != 0):
            return (True, 0)
        return (False, 0)
