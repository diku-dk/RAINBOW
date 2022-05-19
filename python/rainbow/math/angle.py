import numpy as np


def radians_to_degrees(radians):
    return 180.0 * radians / np.pi


def degrees_to_radians(degrees):
    return degrees * np.pi / 180.0
