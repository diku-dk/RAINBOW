import numpy as np


def array_equal(arr1, arr2):
    return None == np.testing.assert_array_almost_equal(arr1, arr2)

def array_not_equal(arr1, arr2):
    return np.any(np.not_equal(arr1,arr2))