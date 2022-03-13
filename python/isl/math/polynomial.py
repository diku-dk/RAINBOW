import math
import numpy as np
import scipy.linalg as linalg


# 2022-03-13 Kenny: TODO Add a Poly1 and Poly3 class


class Helpers:
    """
    This Helper class contains functions that are not intended to be called
    directly by an end-user but are used as building blocks in more complex
    interface functions. It is not recommended using these directly.
    """

    @staticmethod
    def poly2_number_of_monomial_terms(order: int) -> int:
        """
        Computes the number of terms (monomials) in a two-dimensional polynomial of the given order.

        :param order:   The order/degree of the polynomial.
        :return:        The number of terms.
        """
        if order < 0:
            raise ValueError("Polynomial order must be non-negative")
        return math.comb(order + 2, 2)

    @staticmethod
    def poly2_make_monomial_powers(order: int) -> np.ndarray:
        """
        Generate the powers of all monomials in a two-dimensional polynomial of the given order/degree.

        :param order:  The order/degree of the polynomial.
        :return:        A K-by-2 array where each row encodes the powers of a single monomial
                        and K would be the total number of monomials.
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
    def poly2_make_vandermonde_matrix(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
        """
        Generates the pseudo Vandermonde matrix corresponding to the given
        coordinate input vectors and given order the two-dimensional polynomial
        of the given order.

        :param x:       An array of x-coordinates. Must have the same number of elements as number of monomials.
        :param y:       An array of y-coordinates. Must have the same number of elements as number of monomials.
        :param order:   The order/degree of the polynomial.
        :return:        A square matrix with power-terms of the vectors x and y.
        """
        K = Helpers.poly2_number_of_monomial_terms(order)
        if len(x) < K:
            raise ValueError("Need " + str(K) + " x values to interpolate")
        if len(y) < K:
            raise ValueError("Need " + str(K) + " y values to interpolate")
        V = np.zeros((K, K), dtype=np.float64)
        powers = Helpers.poly2_make_monomial_powers(order)
        for k in range(K):
            for idx, power in enumerate(powers):
                V[k, idx] = x[k]**power[0] * y[k]**power[1]
        return V


class Poly2:

    def __init__(self, order: int) -> None:
        """
        Initialize an instance of Poly2.
        This includes allocating internal arrays for encoding all terms of a two-dimensional polynomial.

        :param order:  The order/degree of the polynomial.
        """
        self.order = order
        self.powers = Helpers.poly2_make_monomial_powers(order)
        self.coefficients = np.ones((self.powers.shape[0],), dtype=np.float64)

    def __str__(self) -> str:
        """
        Convert the Poly2 instance to a string value.

        :return:  The string value for the polynomial.
        """
        return poly2_to_str(self)

    def __call__(self, x: float, y: float) -> float:
        """
        Evaluates the polynomial at the given (x,y) point.

        :param x:  The x-value.
        :param y:  The y-value.
        :return:   The value of the polynomial at position (x,y).
        """
        return poly2_eval(self, x, y)


def poly2_make(order: int, coefficients: np.ndarray) -> Poly2:
    """
    Create 2 variable polynomial of given order.
    One may specify the coefficients of each term in the polynomial.

    :param order:          The desired order (degree) of the polynomial.
    :param coefficients:   The coefficients of the polynomial.
    :return:               The resulting Poly2 instance.
    """
    P = Poly2(order)
    if coefficients is None:
        return P
    if coefficients.shape[0] != P.powers.shape[0]:
        raise ValueError("Number of coefficients must equal number of monomials in polynomial")
    P.coefficients = coefficients.ravel()
    return P


def poly2_interpolate(x: np.ndarray, y: np.ndarray, v: np.ndarray, order:int) -> list[Poly2]:
    """
    Compute a Poly2 instance that interpolates the given input data.
    We want to find coefficients of an interpolating polynomial

     .. math::
        P_n(x,y) = \sum_{\forall 0 \leq i,j \leq n, i+j=n} a_{i,j} x^i y^j

    Let :math:`K` be the number of terms in the polynomial :math:`P_n` then we have
    :math:`K`-coefficients we want to determine.

    The polynomial can be written in matrix vector notation as follows

     .. math::
        P_n(x,y) = [1 x y x^2 y^2 x^3 x^2 y  y^3 \cdots ]  [a_0 a_1 \cdots a_{K-1}]^T

    The idea is that we have :math:`K`-points in the plane and for each point we know the value that we wish to
    interpolate. Hence, we know

     .. math::
        P_n(x_i, y_i) = v_i for i=0 to K-1

    This allows us to form a linear system of :math:`K`-equations.  The coefficient matrix, :math:`V`, is known
    as the generalized Vandermonde matrix, and has this form,

     .. math::
        V = [
            [1 x_0 y_0 x_0^2 y_0^2 x_0^3 x_0^2 y_0  y_0^3 ..... ]
            [1 x_1 y_1 x_1^2 y_1^2 x_1^3 x_1^2 y_1  y_1^3 ..... ]
            ...
            [1_{K-1} x_{K-1} y_{K-1} x_{K-1}^2 y_{K-1}^2 x_{K-1}^3 x_{K-1}^2 y_{K-1}  y_{K-1}^3 ..... ]
        ]

    The values :math:`v_i` forms the right-hand-side vector,

    .. math::
        b = [v_0 v_1 ... v_{K-1}]^^

    and now we can solve the system :math:`V a = b` for the coefficient vector :math:`a`.

    :param x:        An array of x-coordinates.
    :param y:        An array of y-coordinates.
    :param v:        An matrix of values to be interpolated at the given (x,y) points. Each column's gives one
                     unique polynomial.
    :param order:    The order of the resulting polynomials.
    :return:         A list of polynomials that interpolate the given data.
    """
    if order <= 0:
        raise ValueError("Order of polynomial interpolation should be positive")
    K = Helpers.poly2_number_of_monomial_terms(order)
    if len(x) < K:
        raise ValueError("Need " + str(K) + " x values to interpolate")
    if len(y) < K:
        raise ValueError("Need " + str(K) + " y values to interpolate")
    if v.shape[0] < K:
        raise ValueError("Need " + str(K) + " values to interpolate")
    V = Helpers.poly2_make_vandermonde_matrix(x[:K], y[:K], order)
    B = np.array(v[:K])
    A = linalg.solve(V, B)
    Ps = [poly2_make(order, A[:, k]) for k in range(A.shape[1])]
    return Ps


def poly2_eval(Pn: Poly2, x: float, y: float) -> float:
    """
    Evaluate the given polynomial at the given input point.

    :param Pn:  The polynomial to be evaluated.
    :param x:   The x-coordinate.
    :param y:   The y-coordinate.
    :return:    The value of P_n(x,y).
    """
    # 13-03-2022 kenny: TODO extend code such that x and y could be arrays of multiple values.
    return np.sum(np.multiply(Pn.coefficients,  np.multiply(x**Pn.powers[:, 0], y**Pn.powers[:, 1])))


def poly2_to_str(Pn: Poly2) -> str:
    """
    Converts the given polynomial into a string value.

    :param Pn:   The polynomial.
    :return:     A string value that display the polynomial in its canonical form.
    """
    polynomial = ""
    for idx, power in enumerate(Pn.powers):
        term = ""
        sign = "+"
        if Pn.coefficients[idx] < 0:
            sign = "-"
        c = math.fabs(Pn.coefficients[idx])
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


def poly2_derivative(Pn: Poly2) -> Poly2:
    """
    This function generates a polynomial that corresponds to the
    derivative of the given polynomial.

    :param Pn:     The input polynomial.
    :return:       The outout polynomial.
    """
    pass


if __name__ == "__main__":
    for order in range(5):
        Pn = Poly2(order)
        print("Polynomial of order ", Pn.order, " can be written as: ", Pn)
        print("It has value Pn(1,1) = ", Pn(1, 1))
    x = np.random.rand(100, 1)*100
    y = np.random.rand(100, 1)*100
    v = np.random.rand(100, 10)
    P1s = poly2_interpolate(x, y, v, 1)
    for i in range(10):
        print("Interpolating polynomial is = ", P1s[i])
        print("Its value at point (1,1) is = ", P1s[i](1, 1))
