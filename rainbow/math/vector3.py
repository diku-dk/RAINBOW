import numpy as np
import rainbow.util.parse_string as parse


def zero():
    return np.zeros((3,), dtype=np.float64)


def ones():
    return np.ones((3,), dtype=np.float64)


def make(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def make_vec4(x, y, z, w):
    return np.array([x, y, z, w], dtype=np.float64)

def check_string(lower_value):
    """
        Parsing a string. The function returns true if the given
        input string follows one of the above templates

        :param: A text string, that consist of valid oprations.
        :return: A boolean telling if the string is valid.
    """
    if parse.parse_string_to_array_check(lower_value):
        return True
    return parse.parse_string_to_random_range_check(lower_value)

def from_string(value):
    value_lower = value.lower()

    if value_lower == "ones":
        return ones()

    if value_lower == "zero":
        return zero()

    if value_lower == "i":
        return i()

    if value_lower == "j":
        return j()

    if value_lower == "k":
        return k()

    assert check_string(value_lower)

    if value_lower.startswith("rand:"):
        (lower_str, upper_str) = value_lower.strip("rand:").split(":")
        return rand(float(lower_str), float(upper_str))
    
    string_2_array =  np.fromstring(value_lower.strip("[]"), dtype=np.float64, sep=",")

    assert len(string_2_array) >= 3

    return string_2_array[:3]


def i():
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def j():
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def k():
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def make_orthonormal_vectors(n):
    """
    This function is used to generate orthonormal vectors. It is given one
    input vector (assumed a normal vector) and then it generates two other
    vector: a tangent vector t and a binormal vector b.

    :param n:  The input normal vector.
    :return:   The triplet of t, b, and n vectors as output.
    """
    # First we make sure we have a unit-normal vector.
    n = np.copy(n)
    n /= np.linalg.norm(n)
    # Next we try to find a direction that is sufficiently different
    # from the normal vector n. We use the coordinate axis mostly
    # pointing away from the n-vector.
    #
    # We will use this unit-direction as guess for a tangent
    # direction.
    [nx, ny ,nz] = np.fabs(n)
    if nx <= ny and nx <= nz:
        t = i()
    if ny <= nx and ny <= nz:
        t = j()
    if nz <= nx and nz <= ny:
        t = k()
    # We now generate a binormal vector, we know
    #
    #   n = t x b
    #   t = b x n
    #   b = n x t
    #
    # We idea is simply to use the t-vector we guessed at to generate
    # a vector we know will be orthonormal to the n-vector.
    b = np.cross(n, t, axis=0)
    b /= np.linalg.norm(b)
    # Now we know that n and be are orthonormal vectors, we we
    # can now compute a third orthonormal t-vector.
    t = np.cross(b, n, axis=0)
    return t, b, n


def cross(a, b):
    return np.cross(a, b, axis=0)


def unit(a):
    return a / np.linalg.norm(a)

def norm(a):                 #pragma: no cover
    return np.linalg.norm(a) #pragma: no cover 


def max_abs_component(a):
    b = np.fabs(a)
    if b[0] > b[1] and b[0] > b[2]:
        return 0
    if b[1] > b[2]:
        return 1
    return 2


def rand(lower, upper):
    return np.random.uniform(lower, upper, 3)


def less(a, b):
    if a[0] > b[0]:
        return False
    if a[0] < b[0]:
        return True
    # We know now that a0==b0
    if a[1] > b[1]:
        return False
    if a[1] < b[1]:
        return True
    # We know not that a0==b0 and a1==b1
    if a[2] > b[2]:
        return False
    if a[2] < b[2]:
        return True
    # We know now that a0==b0 and a1==b1 and a2==b2
    return False

'''
    Spelling error? 
    greather or greater
'''
def greather(a, b):
    if a[0] > b[0]:
        return True
    if a[0] < b[0]:
        return False
    # We know now that a0==b0
    if a[1] > b[1]:
        return True
    if a[1] < b[1]:
        return False
    # We know not that a0==b0 and a1==b1
    if a[2] > b[2]:
        return True
    if a[2] < b[2]:
        return False
    # We know now that a0==b0 and a1==b1 and a2==b2
    return False


def less_than_equal(a, b):
    return not greather(a, b)


def greather_than_equal(a, b):
    return not less(a, b)


if __name__ == "__main__":                                 # pragma: no cover         
    print(max_abs_component(np.array([1.0, 0.0, 0.0],)))   # pragma: no cover
    print(max_abs_component(np.array([-1.0, 0.0, 0.0],)))  # pragma: no cover
    print(max_abs_component(np.array([0.0, 1.0, 0.0],)))   # pragma: no cover
    print(max_abs_component(np.array([0.0, -1.0, 0.0],)))  # pragma: no cover
    print(max_abs_component(np.array([0.0, 0.0, 1.0],)))   # pragma: no cover
    print(max_abs_component(np.array([0.0, 0.0, -1.0],)))  # pragma: no cover

    print(max_abs_component(np.array([1.0, 0.5, 0.1],)))   # pragma: no cover
    print(max_abs_component(np.array([-1.0, 0.5, 0.1],)))  # pragma: no cover
    print(max_abs_component(np.array([0.5, 1.0, 0.1],)))   # pragma: no cover
    print(max_abs_component(np.array([0.5, -1.0, 0.1],)))  # pragma: no cover
    print(max_abs_component(np.array([0.1, 0.5, 1.0],)))   # pragma: no cover
    print(max_abs_component(np.array([-0.1, 0.5, -1.0],))) # pragma: no cover

    print(max_abs_component(np.array([1.0, 0.0, 1.0],)))   # pragma: no cover
    print(max_abs_component(np.array([-1.0, 0.0, -1.0],))) # pragma: no cover
    print(max_abs_component(np.array([0.0, 1.0, 1.0],)))   # pragma: no cover
    print(max_abs_component(np.array([0.0, -1.0, -1.0],))) # pragma: no cover
    print(max_abs_component(np.array([1.0, 1.0, 0.0],)))   # pragma: no cover
    print(max_abs_component(np.array([-1.0, -1.0, 0.0],))) # pragma: no cover
    print(max_abs_component(np.array([1.0, 1.0, 1.0],)))   # pragma: no cover
    print(max_abs_component(np.array([-1.0, -1.0, 1.0],))) # pragma: no cover
