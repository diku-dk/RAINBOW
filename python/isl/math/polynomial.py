import math
import numpy as np
import scipy.linalg as linalg


# TODO Add a Poly1D and Poly3D class


class Helpers:

    @staticmethod
    def poly2d_number_of_monomial_terms(order: int):
        """

        :param order:
        :return:
        """
        return math.comb(order + 2, 2)

    @staticmethod
    def poly2d_make_monomial_powers(order: int):
        """
        :param order:  The degree of the polynomial
        :return:        A K-by-2 array where each row encodes the powers of a monomial and K would be the total number of monomials.
        """
        patterns = []
        if order < 0:
            raise ValueError("Polynomial degree must be non-negative")
        for p in range(order + 1):
            for j in range(p + 1):
                i = p - j
                patterns.append([i, j])
        powers = np.array(patterns, dtype=int)
        return powers

    @staticmethod
    def poly2d_make_vandermonde_matrix(x, y, order: int):
        """

        :param x:
        :param y:
        :param order:
        :return:
        """
        K = Helpers.poly2d_number_of_monomial_terms(order)
        V = np.zeros((K, K), dtype=np.float64)
        powers = Helpers.poly2d_make_monomial_powers(order)
        for k in range(K):
            for idx, power in enumerate(powers):
                V[k, idx] = x[k]**power[0] * y[k]**power[1]
        return V


class Poly2D:

    def __init__(self, order: int):
        """

        :param order:
        """
        self.order = order
        self.powers = Helpers.poly2d_make_monomial_powers(order)
        self.coefficients = np.ones((self.powers.shape[0],), dtype=np.float64)

    def __str__(self):
        """

        :return:
        """
        polynomial = ""
        for idx, power in enumerate(self.powers):
            term = ""
            sign = "+"
            if self.coefficients[idx] < 0:
                sign = "-"
            c = math.fabs(self.coefficients[idx])
            if c > 0:
                if c.is_integer():
                    coef = str(int(c))
                else:
                    coef = str(c)
                if power[0] == 0 and power[1] == 0:
                    term = coef
                if power[0] != 0 and power[1] == 0:
                    term = coef + " x^" + str(power[0])
                if power[0] == 0 and power[1] != 0:
                    term = coef + " y^" + str(power[1])
                if power[0] != 0 and power[1] != 0:
                    term = coef + " x^" + str(power[0]) + " y^" + str(power[1])
            if len(polynomial) > 0:
                polynomial += " " + sign + " " + term
            else:
                polynomial = term
        return polynomial

    def __call__(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        # TODO extend code such that x and y could be arrays of mutliple values.
        return np.sum(np.multiply(self.coefficients,  np.multiply(x**self.powers[:, 0], y**self.powers[:, 1])))


def interpolate_poly2D(x, y, v, order):
    """
    We want to find coefficients of an interpolating polynomial

        P_n(x,y) = sum_{\forall 0 \leq i,j \leq n, i+j=n} a_{i,j} x^i y^j

    Let K be the number of terms in the polynomial P_n then we have
    K-coefficients we want to determine.

    The polynomial can be written in matrix vector notation as follows

        P_n(x,y) = [1 x y x^2 y^2 x^3 x^2 y  y^3 ..... ]  [a_0 a_1 ... a_{K-1}]^T

    The idea is that we have K-points in the plane and for each point we know the value that we wish to
    interpolate. Hence, we know

        P_n(x_i, y_i) = v_i for i=0 to K-1

    This allows us to form a linear system of K equations.  The coefficient matrix, V, is known
    as the generalized Vandermonde matrix, and has this form,

    V = [
    [1 x_0 y_0 x_0^2 y_0^2 x_0^3 x_0^2 y_0  y_0^3 ..... ]
    [1 x_1 y_1 x_1^2 y_1^2 x_1^3 x_1^2 y_1  y_1^3 ..... ]
    ...
    [1_{K-1} x_{K-1} y_{K-1} x_{K-1}^2 y_{K-1}^2 x_{K-1}^3 x_{K-1}^2 y_{K-1}  y_{K-1}^3 ..... ]
    ]

    The values v_i forms the right-hand-side vector,

    b = [v_0 v_1 ... v_{K-1}]^^

    and now we can solve the system V a = b for the coefficient vector a.
    """
    if order <= 0:
        raise ValueError("Order of polynomial interpolation should be positive")
    Pn = Poly2D(order)
    K = Helpers.poly2d_number_of_monomial_terms(order)
    if len(x) < K:
        raise ValueError("Need "+str(K) + " x values to interpolate")
    if len(y) < K:
        raise ValueError("Need "+str(K) + " y values to interpolate")
    if len(v) < K:
        raise ValueError("Need "+str(K) + " values to interpolate")
    V = Helpers.poly2d_make_vandermonde_matrix(x[:K], y[:K], order)
    # TODO Extend code such that v could be a matrix. Hence we want to fine more than one interpolating polynomial.
    b = np.array(v[:K])
    Pn.coefficients = linalg.solve(V, b).ravel()
    return Pn


if __name__ == "__main__":
    for order in range(5):
        Pn = Poly2D(order)
        print("Polynomial of order ", Pn.order, " can be written as: ", Pn)
        print("It has value Pn(1,1) = ", Pn(1, 1))
    x = np.random.rand(100, 1)*100
    y = np.random.rand(100, 1)*100
    v = np.random.rand(100, 1)
    P1 = interpolate_poly2D(x, y, v, 1)
    print("Interpolating polynomial is = ", P1)
    print("Its value at point (1,1) is = ", P1(1, 1))
