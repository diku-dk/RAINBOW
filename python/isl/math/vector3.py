import numpy as np


def zero():
    return np.zeros((3, ), dtype=np.float64)


def ones():
    return np.ones((3,), dtype=np.float64)


def make(x, y, z):
    return np.array([x, y, z], dtype=np.float64)


def make_vec4(x, y, z, w):
    return np.array([x, y, z, w], dtype=np.float64)


def from_string(value):
    if value == 'ones':
        return ones()

    if value == 'zero':
        return zero()

    if value == 'i':
        return i()

    if value == 'j':
        return j()

    if value == 'k':
        return k()

    if value.startswith('rand:'):
        (lower_str, upper_str) = value.strip('rand:').split(':')
        return rand(float(lower_str), float(upper_str))

    return np.fromstring(value.strip('[]'), dtype=np.float64, count=3, sep=',')


def i():
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def j():
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


def k():
    return np.array([0.0, 0.0, 1.0], dtype=np.float64)


def make_orthonormal_vectors(n):
    tmp = np.fabs(n)
    if tmp[0] > tmp[1]:
        if tmp[0] > tmp[2]:
            tx = j()
        else:
            tx = i()
    else:
        if tmp[1] > tmp[2]:
            tx = k()
        else:
            tx = i()
    ty = np.cross(n, tx, axis=0)
    ty /= np.linalg.norm(ty)
    tx = np.cross(ty, n, axis=0)
    return tx, ty, n


def cross(a, b):
    return np.cross(a, b, axis=0)


def unit(a):
    return a / np.linalg.norm(a)


def norm(a):
    return np.linalg.norm(a)


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


if __name__ == '__main__':
    print(max_abs_component(np.array([1.0, 0.0, 0.0], )))
    print(max_abs_component(np.array([-1.0, 0.0, 0.0], )))
    print(max_abs_component( np.array([0.0, 1.0, 0.0], )))
    print(max_abs_component(np.array([0.0, -1.0, 0.0], )))
    print(max_abs_component( np.array([0.0, 0.0, 1.0], )))
    print(max_abs_component(np.array([0.0, 0.0, -1.0], )))

    print(max_abs_component(np.array([1.0, 0.5, 0.1], )))
    print(max_abs_component(np.array([-1.0, 0.5, 0.1], )))
    print(max_abs_component(np.array([0.5, 1.0, 0.1], )))
    print(max_abs_component(np.array([0.5, -1.0, 0.1], )))
    print(max_abs_component(np.array([0.1, 0.5, 1.0], )))
    print(max_abs_component(np.array([-0.1, 0.5, -1.0], )))

    print(max_abs_component(np.array([1.0, 0.0, 1.0], )))
    print(max_abs_component(np.array([-1.0, 0.0, -1.0], )))
    print(max_abs_component(np.array([0.0, 1.0, 1.0], )))
    print(max_abs_component(np.array([0.0, -1.0, -1.0], )))
    print(max_abs_component(np.array([1.0, 1.0, 0.0], )))
    print(max_abs_component(np.array([-1.0, -1.0, 0.0], )))
    print(max_abs_component(np.array([1.0, 1.0, 1.0], )))
    print(max_abs_component(np.array([-1.0, -1.0, 1.0], )))
