import numpy as np

"""
This is a set of extensions to the Python standard library’s unit testing framework. The extensions are mostly
concerned with making it convenient to test array-like data types.
"""


def is_array_equal(arr1, arr2, dec=8):
    """
    Checks if two arrays are almost equal by comparing each element up to a specified decimal place.

    :param arr1: First input array to compare.
    :param arr2: Second input array to compare against the first array.
    :param dec: The desired precision in decimal places. Default is 8.
    :return: True if the arrays are almost equal, False otherwise.
    """
    return None == np.testing.assert_allclose(arr1, arr2, rtol=10**-dec, atol=10**-dec)


def is_array_not_equal(arr1, arr2):
    """
    2022-04-06 Kenny TODO: Write proper documentation.

    :param arr1:
    :param arr2:
    :return:
    """
    return np.any(np.not_equal(arr1, arr2))


def psu_rand_array_gen(size, min_lim=10, max_lim=100):
    """
    2022-04-06 Kenny TODO: Write proper documentation.

    :param size:
    :param min_lim:
    :param max_lim:
    :return:
    """
    rows, cols = size
    length = rows * cols
    min_value = 10 * np.random.random_sample()
    max_value = min_value * np.random.random_sample()
    values = np.linspace(min_value, max_value, length)
    np.random.shuffle(values)
    return values.reshape(size)


def get_base_folder() -> str:
    """
    Retrieves the folder path to the top-most folder of the code-base.

    :return: The top-folder path.
    """
    import os
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/../.."
