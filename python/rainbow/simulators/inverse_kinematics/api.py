import rainbow.math.vector3 as V3
import rainbow.math.quaternion as Q
from rainbow.simulators.inverse_kinematics.types import *
from rainbow.math.angle import *
#import numpy as np
#from numba import njit
import random
from timeit import default_timer

def __is_valid_euler_code(euler_code):
    """
    Function to test if the Euler angle convention encoded as a string value is supported by this simulator.

    :param euler_code:   A text encoding of an euler angle encoding.
    :return:             If the Euler code is usable then the return value is true otherwise it is false.
    """
    proper_euler_anlges = ['XZX', 'XYX', 'YXY', 'YZY', 'ZYZ', 'ZXZ']
    tait_bryan_angles = ['XZY', 'XYZ', 'YXZ', 'YZX', 'ZYX', 'ZXY']
    if euler_code in proper_euler_anlges:
        return True
    if euler_code in tait_bryan_angles:
        return True
    return False


def create_skeleton():
    """
    This is a factory function for creating an empty IK skeleton.

    :return:  The skeleton that was created.
    """
    S = Skeleton()
    return S

def shiftDegToRad(t0):
#    if t0 > 180:
#        return (np.pi/180)*(t0-360)
#    return (np.pi/180)*(t0-180) 
    return (np.pi/180)*(t0) 

def set_chain_length(chains, lengths):
    """
    This functions shears the chains with the specified lengths. A chain contains
    an array of a bone-indices. When shearing, we keep the last bone n in the array
    until we reach the n-length bone. We are taking the last bone, because that
    is where the end-effector-bone is.

    :param chains:        The chains of a skeleton.
    :param lengths:       The alpha Euler angle of the joint that is represented by this bone.
    """
    if len(chains) != len(lengths):
        raise RuntimeError('set_chain_length(): Internal error, chains-array is misaligned with lengths-array')
    for i in range(len(chains)):
        tmpArr = []
        chainLen = lengths[i]-len(chains[i].bones)
        for j in range(lengths[i]):
            tmpArr.append(chains[i].bones[j+chainLen])
        chains[i].bones = np.copy(tmpArr)
        

def create_root(skeleton, alpha, beta, gamma, tx, ty, tz, alphaLim, betaLim, gammaLim, euler_code='ZYZ'):
    """
    This function creates a root bone in an empty skeleton. It should be the first function to use
    when populating a skeleton with bones. A skeleton can only have one root bone.

    :param skeleton:      The skeleton where to create the root bone.
    :param alpha:         The alpha Euler angle of the joint that is represented by this bone.
    :param beta:          The beta Euler angle of the joint that is represented by this bone
    :param gamma:         The gamma Euler angle of the joint that is represented by this bone
    :param tx:            The x-component of the bone vector.
    :param ty:            The y-component of the bone vector.
    :param tz:            The z-component of the bone vector.
    :param alphaLim:      A pair, containing the alpha minimum and maximum limit
    :param betaLim:       A pair, containing the beta minimum and maximum limit
    :param gammaLim:      A pair, containing the gamma minimum and maximum limit
    :param euler_code:    The Euler angle convention used for the joint represented by this bone. This
                          encodes the meaning of the alpha, beta and gamma joint angle values.
    :return:              A reference to the root bone that was created.
    """
    if skeleton.has_root():
        raise RuntimeError('create_root(): Internal error, root already existed')
    if not __is_valid_euler_code(euler_code):
        raise RuntimeError('create_root(): Internal error, invalid Euler angle code given')
    root = Bone()
    root.t = V3.make(tx, ty, tz)
    root.alpha = alpha
    root.beta = beta
    root.gamma = gamma
    root.euler_code = euler_code
    root.alphaMin = shiftDegToRad(alphaLim[0])
    root.alphaMax = shiftDegToRad(alphaLim[1])
    root.betaMin = shiftDegToRad(betaLim[0])
    root.betaMax = shiftDegToRad(betaLim[1])
    root.gammaMin = shiftDegToRad(gammaLim[0])
    root.gammaMax = shiftDegToRad(gammaLim[1])
    skeleton.bones.append(root)
    return root

def add_bone(skeleton, parent_idx, alpha, beta, gamma, tx, ty, tz, alphaLim, betaLim, gammaLim, euler_code='ZYZ'):
    """
    This function creates a new bone in a skeleton. It needs the index of an existing parent bone to know where to
    create the new bone.

    :param skeleton:      The skeleton where to create the new bone into.
    :param parent_idx:    The index of the parent bone.
    :param alpha:         The alpha Euler angle of the joint that is represented by this bone.
    :param beta:          The beta Euler angle of the joint that is represented by this bone
    :param gamma:         The gamma Euler angle of the joint that is represented by this bone
    :param tx:            The x-component of the bone vector.
    :param ty:            The y-component of the bone vector.
    :param tz:            The z-component of the bone vector.
    :param alphaLim:      A pair, containing the alpha minimum and maximum limit
    :param betaLim:       A pair, containing the beta minimum and maximum limit
    :param gammaLim:      A pair, containing the gamma minimum and maximum limit
    :param euler_code:    The Euler angle convention used for the joint represented by this bone. This
                          encodes the meaning of the alpha, beta and gamma joint angle values.
    :return:              A reference to the new bone that was created in the skeleton.
    """
    if not skeleton.has_bone(parent_idx):
        raise RuntimeError('add_bone(): Internal error, parent did not exist')
    if not __is_valid_euler_code(euler_code):
        raise RuntimeError('create_root(): Internal error, invalid Euler angle code given')

    end_effector = Bone()
    end_effector.idx = len(skeleton.bones)
    end_effector.parent = parent_idx
    end_effector.t = V3.make(tx, ty, tz)
    end_effector.alpha = alpha
    end_effector.beta = beta
    end_effector.gamma = gamma
    end_effector.alphaMin = shiftDegToRad(alphaLim[0])
    end_effector.alphaMax = shiftDegToRad(alphaLim[1])
    end_effector.betaMin = shiftDegToRad(betaLim[0])
    end_effector.betaMax = shiftDegToRad(betaLim[1])
    end_effector.gammaMin = shiftDegToRad(gammaLim[0])
    end_effector.gammaMax = shiftDegToRad(gammaLim[1])
    end_effector.euler_code = euler_code

    skeleton.bones[parent_idx].children.append(end_effector.idx)
    skeleton.bones.append(end_effector)
    return end_effector


def __update_bone(bone, skeleton) -> None:
    """
    This function extract the joint angles of the current bone and use these to compute the relative transformation
    from parent bone to this bone. It then extracts the world coordinate pose of the parent bone and use all the
    information to compute the bones own world coordinate pose.

    Once this is done the function update the child bones world coordinate pose by invoking itself recursively.

    Hence, this function recursively traverse the skeleton from root to all end-effectors and while doing so it
    updates the world coordinate system pose of each bone in the skeleton.

    :param bone:          The current bone to update.
    :param skeleton:      The IK skeleton holding all the bones.
    :return:              None.
    """
    q_alpha = bone.get_rotation_alpha()
    q_beta = bone.get_rotation_beta()
    q_gamma = bone.get_rotation_gamma()
    q_bone = Q.prod(q_alpha, Q.prod(q_beta, q_gamma))

    t_parent = V3.zero()
    q_parent = Q.identity()
    if bone.has_parent():
        t_parent = skeleton.bones[bone.parent].t_wcs
        q_parent = skeleton.bones[bone.parent].q_wcs
    bone.t_wcs = t_parent + Q.rotate(q_parent, bone.t)
    bone.q_wcs = Q.prod(q_parent, q_bone)

    for idx in bone.children:
        __update_bone(skeleton.bones[idx], skeleton)


def update_skeleton(skeleton) -> None:
    """
    This function makes sure that all bones in a skeleton get updated to have the correct world
    coordinate system pose.

    One should call this method if one have changed the joint angle values stored in the skeleton and now needs
    to know where things are located in the world coordinate system.

    :param skeleton:  The skeleton that must be updated.
    :return:          None.
    """
    __update_bone(skeleton.bones[0], skeleton)


def __make_chain(end_effector, skeleton):
    """
    This function traverses a skeleton from the given end-effectors. It creates
    a serial chain going from root bone to the given end-effector. Each chain stores the bone indices of all
    bones on the path from the root to the end-effector.

    :param end_effector:   A reference to the bone that one wish to use as an end-effector.
    :param skeleton:       The skeleton which the end-effector bone belongs to.
    :return:               The resulting IK chain.
    """
    chain = Chain()
    chain.skeleton = skeleton
    bone = end_effector
    while not bone.is_root():
        chain.bones.append(bone.idx)
        bone = skeleton.bones[bone.parent]
    chain.bones.append(bone.idx)
    chain.bones.reverse()
    return chain


def make_chains(skeleton):
    """
    A skeleton is basically a tree structure. When manipulating a skeleton with inverse kinematics one does this
    my specifying a goal for a so-called end-effector. An end-effector can be any bone in the skeleton. However,
    end-effectors are often taken to be the leaves of the skeleton tree structure.

    We create this default set of end-effectors by identifying all leaves in the skeleton.

    Because we manipulate the skeleton my specifying goals for our end-effectors we will for each end-effector
    store this goal as well as all the bones on the path from the end-effector to the root bone of the
    skeleton. This path of bones makes up a serial chain from root to end-effector. Hence, we create a list
    of chains where each chain cooresponds to one end-effector.

    This wording of "chain" and "end-effector" comes from Robotics. The computer science equivalent would be
    something like "path" and "leave".

    :param skeleton:  The skeleton to create serial chains from.
    :return:          A list of all the default serial chains of the skeleton.
    """
    chains = []
    for bone in skeleton.bones:
        if bone.is_end_effector():
            chain = __make_chain(bone, skeleton)
            chains.append(chain)
    return chains

def print_chains(chains):
    for chain in chains:
        print('chain:')
        for idx in chain.bones:
            print('\t', idx)
        print()

def get_joint_angles(skeleton):
    """
    This function retrieves all the joint angles from an IK-skeleton and returns them as a
    numpy array. This is convenient for converting from skeleton to numpy. The reverse conversion
    can be done by using the function set_joint_angles.

    :param skeleton:  The IK skeleton from where to retrieve the joint angles.
    :return:          A numpy array holding the joint angle values.
    """
    angles = np.zeros((len(skeleton.bones) * 3,), dtype=np.float64)
    for bone in skeleton.bones:
        angles[bone.idx * 3 + 0] = bone.alpha
        angles[bone.idx * 3 + 1] = bone.beta
        angles[bone.idx * 3 + 2] = bone.gamma
    return angles


def set_joint_angles(skeleton, angles) -> None:
    """
    This is a convenience function that allow one to use a whole numpy array of joint angle values to
    set the joint angles stored in a skeleton data structure. This is nice for converting from numpy arrays
    to skeleton data structure.

    :param skeleton:       The IK skeleton where to set the joint angles.
    :param angles:         The numpy array that stores the joint angle values to use.
    :return:               None.
    """
    for bone in skeleton.bones:
        bone.alpha = angles[bone.idx * 3 + 0]
        bone.beta = angles[bone.idx * 3 + 1]
        bone.gamma = angles[bone.idx * 3 + 2]


def get_end_effector(chain, skeleton):
    """
    This function retrieve the end-effector position of a chain in the world coordinate system (wcs). The function
     assumes the skeleton pose has been updated prior to invocation by using the function update_skeleton.

    :param chain:       The IK chain which the end-effector belongs to.
    :param skeleton:    The IK skeleton, assumed to have been updated by invoking update_skeleton.
    :return:            The 3D position of the end-effector in the world coordinate system.
    """
    t_wcs = skeleton.bones[chain.bones[-1]].t_wcs
    q_wcs = skeleton.bones[chain.bones[-1]].q_wcs
    tool = chain.tool
    return t_wcs + Q.rotate(q_wcs, tool)
    
def print_jacobian_old(J):
    """
    This function is for debug purposes only and is used to print the Jacobian
    more human readable form.
    
    :param chain:       The Jacobian to print.
    """
    print("Dimensions jacobian: " + str(J.shape) + ".")
    print("Contents of jacobian matrix:")
    string = "["
    for i in range(J.shape[0]):
        string += "["
        for j in range(J.shape[1]):
            string += str(J[i][j]) + ", "
            if ((j+1) % 3 == 0):
                string += "],\n"
        string += "],\n"
    string += "]"
    print(string)

def print_jacobian_old(J):
    """
    This function is for debug purposes only and is used to print the Jacobian
    more human readable form.
    
    :param chain:       The Jacobian to print.
    """
    print("Dimensions jacobian: " + str(J.shape) + ".")
    print("Contents of jacobian matrix:")
    string = "["
    for i in range(J.shape[0]):
        string += "["
        for j in range(J.shape[1]):
            string += str(f'{J[i][j]:.6f}') + ", "
        string += "],\n"
    string += "]"
    print(string)

def print_jacobian(J):
    """
    This function is for debug purposes only and is used to print the Jacobian
    more human readable form.
    
    :param chain:       The Jacobian to print.
    """
    print("Dimensions jacobian: " + str(J.shape) + ".")
    print("Contents of jacobian matrix:")
    string = ""
    for i in range(int(J.shape[0]/3)):
        string += "Row " + str(i) + ": \n"
        for t in range(int(J.shape[1]/3)):
            string += "Column " + str(t) + ": [["
            for j in range(3):
                if j == 2:
                    string += str(f'{J[i*3+1][j+t*3]:.6f}')
                else:
                    string += str(f'{J[i*3+1][j+t*3]:.6f}') + ", "
            string += "],["
            for k in range(3):
                if k == 2:
                    string += str(f'{J[i*3+1][k+t*3]:.6f}')
                else:
                    string += str(f'{J[i*3+1][k+t*3]:.6f}') + ", "
            string += "],["
            for l in range(3):
                if l == 2:
                    string += str(f'{J[i*3+1][l+t*3]:.6f}')
                else:
                    string += str(f'{J[i*3+1][l+t*3]:.6f}') + ", "
            string += "]],\n"
        
        string += ",\n\n\n"
    string += "]"
    print(string)


def normalize(a):
    vn = np.linalg.norm(a)
    if vn == 0:
        return a
    return a/vn

def compute_jacobian(chains, skeleton):
    """
    Compute the Jacobian of the forward kinematic function. That is the function that computes the
    end-effector positions in world-space given the current joint-angle values.

    :param chains:    The IK chains defining the end-effectors.
    :param skeleton:  The IK skeleton.
    :return:          A numpy matrix with the value of the Jacobian.
    """
    num_angles = len(skeleton.bones) * 3
    num_coords = len(chains) * 3

    J = np.zeros((num_coords, num_angles), dtype=np.float64)
    row_offset = 0
    for chain in chains:
        e = get_end_effector(chain, skeleton)
        for idx in chain.bones:
            bone = skeleton.bones[idx]
            q_parent = Q.identity()
            if bone.has_parent():
                q_parent = skeleton.bones[bone.parent].q_wcs

            # In the 5 lines of code below the ZYZ Euler angles are hardwired into the code.
            q_alpha = bone.get_rotation_alpha()
            q_alpha_beta = Q.prod(q_alpha, bone.get_rotation_beta())

            u = Q.rotate(q_parent, bone.get_axis_alpha())
            v = Q.rotate(q_parent, Q.rotate(q_alpha, bone.get_axis_beta()))
            w = Q.rotate(q_parent, Q.rotate(q_alpha_beta, bone.get_axis_gamma()))

            delta_p = e - bone.t_wcs
            J_alpha = V3.cross(u, delta_p)
            J_beta = V3.cross(v, delta_p)
            J_gamma = V3.cross(w, delta_p)

            J[row_offset: row_offset + 3, idx * 3 + 0] = J_alpha
            J[row_offset: row_offset + 3, idx * 3 + 1] = J_beta
            J[row_offset: row_offset + 3, idx * 3 + 2] = J_gamma
        row_offset += 3
    return J


def compute_hessian(chains, skeleton, J):
    """
    This function compute the Hessian of the IK objective function wrt all the joint angles.

    This implementation is based on an analytical solution for computing the gradient and is hence much faster than
    a numerical approximation method.

    :param chains:       The IK chains of the IK problem.
    :param skeleton:     The IK skeleton (basically storing the joint angle values)
    :param J:            The Jacobian matrix of the forward kinematics functions. Should be computed
                         using the function compute_jacobian.
    :return:             A numpy matrix holding the Hessian value.
    """
    H = np.dot(np.transpose(J), J)
    row_offset = 0
    for chain in chains:
        e = get_end_effector(chain, skeleton)
        r = chain.goal - e
        for k in chain.bones:
            for h in chain.bones:
                if h > k:
                    continue

                Bh = skeleton.bones[h]

                q_parent = Q.identity()
                if Bh.has_parent():
                    q_parent = skeleton.bones[Bh.parent].q_wcs

                # In the 7 lines of code below the ZYZ Euler angles are hardwired into the code.
                q_alpha = Bh.get_rotation_alpha()
                q_beta = Bh.get_rotation_beta()
                q_gamma = Bh.get_rotation_gamma()
                q_alpha_beta = Q.prod(q_alpha, q_beta)

                u = Q.rotate(q_parent, Bh.get_axis_alpha())
                v = Q.rotate(q_parent, Q.rotate(q_alpha, Bh.get_axis_beta()))
                w = Q.rotate(q_parent, Q.rotate(q_alpha_beta, Bh.get_axis_gamma()))

                k_offset = k * 3
                h_offset = h * 3

                J_a = J[row_offset:row_offset + 3, k_offset]
                J_b = J[row_offset:row_offset + 3, k_offset + 1]
                J_c = J[row_offset:row_offset + 3, k_offset + 2]

                ua = np.dot(V3.cross(u, J_a), r)
                va = np.dot(V3.cross(v, J_a), r)
                wa = np.dot(V3.cross(w, J_a), r)

                ub = np.dot(V3.cross(u, J_b), r)
                vb = np.dot(V3.cross(v, J_b), r)
                wb = np.dot(V3.cross(w, J_b), r)

                uc = np.dot(V3.cross(u, J_c), r)
                vc = np.dot(V3.cross(v, J_c), r)
                wc = np.dot(V3.cross(w, J_c), r)

                dH = np.array(
                    [[ua, va, wa],
                     [ub, vb, wb],
                     [uc, vc, wc]]
                )

                H[h_offset:h_offset + 3, k_offset:k_offset + 3] -= dH
                if h != k:
                    H[k_offset:k_offset + 3, h_offset:h_offset + 3] -= np.transpose(dH)

        row_offset += 3
    return H


def compute_gradient(chains, skeleton, J):
    """
    This function compute the gradient of the IK objective function wrt all the joint angles.

    This implementation is based on an analytical solution for computing the gradient and is hence much faster than
    a numerical approximation method.

    :param chains:       The IK chains of the IK problem.
    :param skeleton:     The IK skeleton (basically storing the joint angle values)
    :param J:            The Jacobian matrix of the forward kinematics functions. Should be computed
                         using the function compute_jacobian.
    :return:             A numpy array holding the gradient value.
    """
    r = np.zeros((len(chains) * 3,), dtype=np.float64)
    row_offset = 0
    for chain in chains:
        e = get_end_effector(chain, skeleton)
        
        r[row_offset:row_offset+3] = chain.goal - e
        row_offset += 3
    
    g = -np.dot(np.transpose(J), r)
    return g


def compute_objective(chains, skeleton) -> float:
    """
    This function computes the objective value for the IK problem represented by the IK chains. Each
    chain describe a goal for an end-effector of a serial IK chain. This function essentially
    measure the errors between corresponding goals and end-effectors and sum up all the errors to
    a single float value. A value of zero means all goals are fulfilled, a larger value means larger errors. Hence,
    to solve the IK problem one wish to minimize the objective function value as much as possible.

    :param chains:    The IK chains of the skeleton.
    :param skeleton:  The IK skeleton.
    :return:          The value of the IK objective function.
    """
    f = 0
    for chain in chains:
        e = get_end_effector(chain, skeleton)
        r = chain.goal - e
        f += np.dot(r, r)
    return f * 0.5


def set_angle(idx, value, skeleton) -> None:
    """
    This function is a convenience function that sets the value of the idx-joint angle. This
    is such that end-users do not have to know about the data structures used.

    :param idx:        The index value of the joint angle (aka variable) that one which to set the value of.
    :param value:      The new value of the idx-joint-angle.
    :param skeleton:   The IK skeleton from which to set the angle value from.
    :return:           None
    """
    k = idx // 3
    offset = idx % 3
    if offset == 0:
        skeleton.bones[k].alpha = value
    elif offset == 1:
        skeleton.bones[k].beta = value
    elif offset == 2:
        skeleton.bones[k].gamma = value
    else:
        raise RuntimeError('set_angle(): no such offset exist')


def get_angle(idx, skeleton) -> float:
    """
    This function is a convenience function that extract the value of the idx-joint angle. This
    is such that end-users do not have to know about the data structures used.

    :param idx:        The index value of the joint angle (aka variable) that one which to retrive the value of.
    :param skeleton:   The IK skeleton from which to retrive the angle value from.
    :return:           The value of the idx-joint-angle.
    """
    k = idx // 3
    offset = idx % 3
    if offset == 0:
        return skeleton.bones[k].alpha
    elif offset == 1:
        return skeleton.bones[k].beta
    elif offset == 2:
        return skeleton.bones[k].gamma
    else:
        raise RuntimeError('get_angle(): no such offset exist')


def __numerical_differentiation_second_derivative(chains, skeleton, i, j, h):
    """
    Numerically evaluate the second order derivative of the objective function wrt the i and j joint angles.
    This method is not intended to be invoked directly by end-users.

    :param chains:         The IK chains of the IK skeleton.
    :param skeleton:       The IK skeleton.
    :param i:              The index of the first component (ie angle) that we compute the derivative with respect to.
    :param j:              The index of the second component (ie angle) that we compute the derivative with respect to.
    :param h:              The step-size to use for the finite difference approximation. We use a
                           central difference approximation.
    :return:               The second derivative of the IK objective function wrt variables i and j.
    """
    if i == j:
        update_skeleton(skeleton)
        f_i = compute_objective(chains, skeleton)
        theta_i = get_angle(i, skeleton)
        set_angle(i, theta_i + h, skeleton)
        update_skeleton(skeleton)
        f_iph = compute_objective(chains, skeleton)
        set_angle(i, theta_i - h, skeleton)
        update_skeleton(skeleton)
        f_imh = compute_objective(chains, skeleton)
        set_angle(i, theta_i, skeleton)
        return (f_iph - 2 * f_i + f_imh) / (h * h)
    else:
        theta_i = get_angle(i, skeleton)
        theta_j = get_angle(j, skeleton)
        set_angle(i, theta_i + h, skeleton)
        set_angle(j, theta_j + h, skeleton)
        update_skeleton(skeleton)
        f_iph_jph = compute_objective(chains, skeleton)
        set_angle(i, theta_i - h, skeleton)
        set_angle(j, theta_j + h, skeleton)
        update_skeleton(skeleton)
        f_imh_jph = compute_objective(chains, skeleton)
        set_angle(i, theta_i + h, skeleton)
        set_angle(j, theta_j - h, skeleton)
        update_skeleton(skeleton)
        f_iph_jmh = compute_objective(chains, skeleton)
        set_angle(i, theta_i - h, skeleton)
        set_angle(j, theta_j - h, skeleton)
        update_skeleton(skeleton)
        f_imh_jmh = compute_objective(chains, skeleton)
        set_angle(i, theta_i, skeleton)
        set_angle(j, theta_j, skeleton)
        return (f_iph_jph - f_iph_jmh - f_imh_jph + f_imh_jmh) / (4 * h * h)


def __numerical_differentiation_first_derivative(chains, skeleton, i, h):
    """
    Numerically evaluate the derivative of the objective function wrt a single joint angle. This method is not
    intended to be invoked directly by end-users.

    :param chains:         The IK chains of the IK skeleton.
    :param skeleton:       The IK skeleton.
    :param i:              The index of the component (ie angle) that we compute the derivative with respect to.
    :param h:              The step-size to use for the finite difference approximation. We use a
                           central difference approximation.
    :return:               The derivative of the objective function wrt the i'th variable.
    """
    theta_i = get_angle(i, skeleton)
    set_angle(i, theta_i + h, skeleton)
    update_skeleton(skeleton)
    f_iph = compute_objective(chains, skeleton)
    set_angle(i, theta_i - h, skeleton)
    update_skeleton(skeleton)
    f_imh = compute_objective(chains, skeleton)
    set_angle(i, theta_i, skeleton)
    return (f_iph - f_imh) / (2 * h)


def compute_finite_difference_gradient(chains, skeleton, h=0.1):
    """
    This function uses finite difference method to get the gradient of the IK objective function.
    This is a slow method, and it is subject to numerical approximation error. Hence, it is far
    better to use the function compute_gradient instead.

    :param chains:          All the IK chains.
    :param skeleton:        The IK skeleton.
    :param h:               The step-size to use for the finite difference approximation. We
                            use a 1st order central difference approximation. In general as h goes
                            to zero the approximation error will go to zero as well.

    :return:                The gradient.
    """
    N = len(skeleton.bones) * 3
    g = np.zeros((N,), dtype=np.float64)
    for i in range(N):
        g[i] = __numerical_differentiation_first_derivative(chains, skeleton, i, h)
    return g


def compute_finite_difference_hessian(chains, skeleton, h=0.1):
    """
    This function uses finite difference method to get the gradient of the IK objective function.
    This is a slow method, and it is subject to numerical approximation error. Hence, it is far
    better to use the function compute_gradient instead.

    :param chains:          All the IK chains.
    :param skeleton:        The IK skeleton.
    :param h:               The step-size to use for the finite difference approximation. We
                            use a central difference approximation. In general as h goes
                            to zero the approximation error will go to zero as well.

    :return:                The Hessian.
    """
    N = len(skeleton.bones) * 3
    H = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            H[i, j] = __numerical_differentiation_second_derivative(chains, skeleton, i, j, h)
    return H

def compute_f_theta(chains, skeleton):
    """
    This function is used for computing f(theta) and is needed as a seperate
    function, to find a good value of the step_size_alpha. This function assumes
    the skeleton was updated before invocation. 

    :param chains:          All the IK chains.
    :param skeleton:        The IK skeleton.
    
    :return:                f(theta)
    """
    r = np.zeros((len(chains) * 3,), dtype=np.float64)
    row_offset = 0
    for chain in chains:
        e = get_end_effector(chain, skeleton)
        
        r[row_offset:row_offset+3] = chain.goal - e
        row_offset += 3
    
    f = 0.5*np.dot(np.transpose(r), r)
    return f

def compareArrays(arr1, arr2):
    for i in range(len(arr1)):
        if (abs(arr1[i])-abs(arr2[i]) > 0.0001):
            return False
    return True
    
def compute_step_size_alpha(chains, skeleton, gradient, thetaLast, ftheta, step_size_alpha, rho, c):
    """
    TODO PROPER DOCUMENT"""
    loop = True
    p = -gradient
    deltaFk = np.dot(np.transpose(gradient), p)
    while (loop):
        #Not a perfect solution, but in order to obtain the end-effector, we
        #need to update theta, then update the skeleton, to be able to 
        #extract f(theta).
        #f(x_k + αp_k)
        set_joint_angles(skeleton, thetaLast + step_size_alpha*p)
        update_skeleton(skeleton)
        #f(x_k) + cα∇f_k^T p_k
        if compute_f_theta(chains, skeleton) > (ftheta + c*step_size_alpha*deltaFk):
            step_size_alpha = rho * step_size_alpha
        else:
            loop = False
    return step_size_alpha

def compute_gradient_descent(chains, skeleton, iterations, step_size_alpha, gamma, epsilon, rho, c):
    """
    This function uses finite difference method to get the gradient of the IK objective function.
    This is a slow method, and it is subject to numerical approximation error. Hence, it is far
    better to use the function compute_gradient instead.

    :param chains:          All the IK chains.
    :param skeleton:        The IK skeleton.
    :param iterations:      The maximum amount of iterations the algorithm can
                            run for.
    :param step_size_alpha: A user-defined step-size, which defines how fast the
                            algorithm should descent
    :param gamma:           A user-defined value, which is used to assert, if 
                            the function is still descending or if it has 
                            reached a minimum or plateaus.
    :param epsilon:         A user-defined value and if the norm of the gradient 
                            is smaller than this value, it is likely the
                            gradient is close to a minimum           
    """


    
    start = default_timer()
    elapsedTime = 0.0

    """update_skeleton(skeleton)
    angles = get_joint_angles(skeleton)
    for i in range(len(skeleton.bones)):
        l = skeleton.bones[i].get_limit_lower()
        u = skeleton.bones[i].get_limit_upper()
        for j in range(len(l)):
             val = angles[i*3+j]

             if (val > math.pi):
                 val = math.fmod(val, math.pi)
             elif (val < -math.pi):
                 val = math.fmod(val, -math.pi)
             angles[i*3+j] = min(u[j], max(l[j], val))
    set_joint_angles(skeleton, angles)"""
    
    update_skeleton(skeleton)
    thetaLast = get_joint_angles(skeleton)
    thetaK = np.zeros((len(thetaLast),), dtype=np.float64)
    objectives = []
    its = 0
    maxLoopIts = 0
    LoopItsObjective = []
    for i in range(iterations):
        its += 1
        Jacobian = compute_jacobian(chains, skeleton)
        gradient = compute_gradient(chains, skeleton, Jacobian)
        
        #Compute STEP_SIZE_ALPHA
        ftheta = compute_f_theta(chains, skeleton)
        alpha = compute_step_size_alpha(chains, skeleton, gradient, thetaLast, ftheta, step_size_alpha, rho, c)
        timeStart = default_timer()

        #Compute the actual gradient descent
        #Apply box constraints to each bone individually.
        thetaTemp = thetaLast - alpha * gradient
        for i in range(len(skeleton.bones)):
            l = skeleton.bones[i].get_limit_lower()
            u = skeleton.bones[i].get_limit_upper()
            for j in range(len(l)):
                 val = thetaTemp[i*3+j]

                 if (val > math.pi):
                     val = math.fmod(val, math.pi)
                 elif (val < -math.pi):
                     val = math.fmod(val, -math.pi)
                 thetaK[i*3+j] = min(u[j], max(l[j], val))


        set_joint_angles(skeleton, thetaK)
        update_skeleton(skeleton)
        objectives.append(compute_objective(chains, skeleton).copy())
        
        #Has the gradient descent reached a minimum, or are we close enough to a minimum?
        if (np.linalg.norm(gradient) < epsilon or (np.linalg.norm(thetaK-thetaLast) < gamma)):
            break
        #Python implicit pointers, makes the copy a requirement.
        thetaLast = np.copy(thetaK)

    """    print("Total iterations for the solver: " + str(its))
    print("Total time elapsed for the IK solver " + str(default_timer() - start))
    print("Total time elapsed for finding step size alpha" + str(elapsedTime))
    print("Value of rho: " + str(rho))
    print("Value of c: " + str(c))"""
    import matplotlib.pyplot as plt
    plt.semilogy(range(1, len(objectives)+1), objectives)
    
    plt.title("Convergence plot for an arm. Constant alpha = 0.001")
    plt.xlabel("Iterations")
    plt.ylabel("Objective")
    #plt.show()

def solveVariables(chains, skeleton, iterations, step_size_alpha, gamma, epsilon, rho, c):
    compute_gradient_descent(chains, skeleton, iterations, step_size_alpha, gamma, epsilon, rho, c)
    
def solve(chains, skeleton):
    compute_gradient_descent(chains, skeleton, 200, 0.37, 0.0001, 0.05, 0.21, 0.0001)
#    compute_gradient_descent(chains, skeleton, 500, 0.0005, 0.0001, 0.0001)
#    compute_gradient_descent(chains, skeleton, 100, 0.005, 0.0001, 0.0001)

