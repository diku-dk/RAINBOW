import rainbow.math.quaternion as Q
import rainbow.math.vector3 as V3
import numpy as np

class Bone:

    def __init__(self):
        self.idx = 0  # Bone index
        self.euler_code = 'ZYZ'  # Euler angle convention
        self.alpha = 0.0  # Euler angles of bone
        self.beta = 0.0
        self.gamma = 0.0
        self.q_wcs = Q.identity()  # Joint frame orientation in WCS
        self.t_wcs = V3.zero()  # Joint origin in WCS
        self.t = V3.zero()  # Joint vector in parent frame
        self.parent = -1  # Index to parent bone or -1 if bone is root
        self.children = []  # Indices to children bones

    def __str__(self):
        """
        Get string representation of a bone,
        :return: A json-inspired string showing the content of the bone.
        """
        txt = "{\n"
        txt = txt + "\t\"idx\" : " + str(self.idx) + ",\n"
        txt = txt + "\t\"parent\" : " + str(self.parent) + ",\n"
        txt = txt + "\t\"euler\" : \"" + self.euler_code + "\",\n"
        txt = txt + "\t\"angles\" : { \"alpha\" : " + str(self.alpha) + \
              ", \"beta\" : " + str(self.beta) + \
              ", \"gamma\" : " + str(self.gamma) + "},\n"
        txt = txt + "\t\"origin\" : " + np.array2string(self.t, separator=', ') + ",\n"
        txt = txt + "\t\"children\" : " + str(self.children) + "\n"
        txt = txt + "}"
        return txt

    def is_root(self):
        if self.parent == -1:
            return True
        return False

    def is_end_effector(self):
        if len(self.children) == 0:
            return True
        return False

    def has_parent(self):
        return not self.is_root()

    def get_axis(self, idx):
        if self.euler_code[idx] == 'X':
            return V3.i()
        if self.euler_code[idx] == 'Y':
            return V3.j()
        if self.euler_code[idx] == 'Z':
            return V3.k()

    def get_rotation(self, idx, radians):
        if self.euler_code[idx] == 'X':
            return Q.Rx(radians)
        if self.euler_code[idx] == 'Y':
            return Q.Ry(radians)
        if self.euler_code[idx] == 'Z':
            return Q.Rz(radians)

    def get_axis_alpha(self):
        return self.get_axis(0)

    def get_axis_beta(self):
        return self.get_axis(1)

    def get_axis_gamma(self):
        return self.get_axis(2)

    def get_rotation_alpha(self):
        return self.get_rotation(0, self.alpha)

    def get_rotation_beta(self):
        return self.get_rotation(1, self.beta)

    def get_rotation_gamma(self):
        return self.get_rotation(2, self.gamma)


class Skeleton:

    def __init__(self):
        self.bones = []  # Bones of the skeleton.

    def __str__(self):
        """
        Get string representation of a skeleton.

        :return: A json inspired string representing the skeleton data.
        """
        list_of_bone_strings = ["\t\"bone"+str(bone.idx)+"\" : " + str(bone).replace("\n", "\n\t") for bone in self.bones]
        txt = "{\n" + ",\n".join(list_of_bone_strings) + "\n}"
        return txt

    def has_root(self):
        if len(self.bones) > 0:
            return True
        return False

    def has_bone(self, idx):
        if len(self.bones) > idx >= 0:
            return True
        return False


class Chain:

    def __init__(self):
        self.bones = []  # Indices of all bones that are part of the chain.
        self.skeleton = None  # Reference to skeleton holding the bones
        self.goal = V3.make(10, 0, 0)  # A default goal position in world coordinates
        self.tool = V3.zero()  # A tool vector in end-effector coordinates

    def __str__(self):
        """
        Get string representation of the chain.

        :return: A json inspired string showing the content of the chain.
        """
        txt = "{\n"
        txt = txt + "\t\"goal\" : " + np.array2string(self.goal, separator=', ') + ",\n"
        txt = txt + "\t\"tool\" : " + np.array2string(self.tool, separator=', ') + ",\n"
        txt = txt + "\t\"bones\" : " + str(self.bones) + "\n"
        txt = txt + "}"
        return txt
