from numpy import pi

def radians_to_degrees(radians):
#    return 180.0 * radians / 3.141592653589793115997963468544185161590576171875
    return 180 * radians / pi


def degrees_to_radians(degrees):
#    return degrees * 3.141592653589793115997963468544185161590576171875 / 180.0
    return degrees * pi / 180
