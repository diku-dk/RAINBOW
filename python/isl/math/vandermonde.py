import numpy as np

def make_poly_str(degree:int):
    """

    :param degree:
    :return:
    """
    if degree < 0:
        return ""
    polynomial = "1"
    for p in range(degree+1):
        for j in range(p+1):
            i = p-j
            mononial = ""
            if i > 0:
                mononial += "x^" + str(i)
            if j > 0:
                mononial += "y^" + str(j)
            if len(mononial) > 0:
                polynomial += " + " + mononial
    return polynomial


if __name__ == "__main__":
    print(make_poly_str(-1))
    print(make_poly_str(0))
    print(make_poly_str(1))
    print(make_poly_str(2))
    print(make_poly_str(3))
