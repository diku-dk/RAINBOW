import numpy as np


def assert_array_equal(arr1, arr2, dec=8):
    """
    2022-04-06 Kenny TODO: Write proper documentation.

    :param arr1:
    :param arr2:
    :param dec:
    :return:
    """
    return None == np.testing.assert_array_almost_equal(arr1, arr2, dec)

def assert_array_not_equal(arr1, arr2):
    """
    2022-04-06 Kenny TODO: Write proper documentation.

    :param arr1:
    :param arr2:
    :return:
    """
    return np.any(np.not_equal(arr1,arr2))

def psu_rand_array_gen(size, min_lim=10,max_lim=100):
    """
    2022-04-06 Kenny TODO: Write proper documentation.

    :param size:
    :param min_lim:
    :param max_lim:
    :return:
    """
    rows, cols = size
    length     = rows * cols
    min_value  = 10 * np.random.random_sample()
    max_value  = min_value * np.random.random_sample()  
    values     = np.linspace(min_value, max_value, length)
    np.random.shuffle(values)
    return  values.reshape(size)
