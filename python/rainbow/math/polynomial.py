import math
import numpy as np
import scipy.linalg as linalg


# 2022-03-13 Kenny: TODO Add one and three dimensional polynomial support.


class Helpers:
    """
    This Helper class contains functions that are not intended to be called
    directly by an end-user but are used as building blocks in more complex
    interface functions. It is not recommended using these directly.
    """

    @staticmethod
    def number_of_monomial_terms(dimension: int, order: int) -> int:
        """
        Computes the number of terms (monomials) for a polynomial of given dimensional and order.

        :param dimension: The dimension of the polynomial.
        :param order:     The order/degree of the polynomial.
        :return:          The number of terms.
        """
        if order < 0:
            raise ValueError("Polynomial order must be non-negative")
        if dimension < 1:
            raise ValueError("Polynomial dimension must be at least one")
        if dimension > 3:
            raise ValueError("Polynomial dimension must be at most three")
        return math.comb(order + dimension, dimension)

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
            raise ValueError("Polynomial order must be non-negative")
        for p in range(order + 1):
            for j in range(p + 1):
                i = p - j
                patterns.append([i, j])
        powers = np.array(patterns, dtype=int)
        return powers

    @staticmethod
    def make_vandermonde_matrix(order: int, x: np.ndarray) -> np.ndarray:
        """
        Generates the pseudo Vandermonde matrix corresponding to the given
        coordinate input vectors and given order the two-dimensional polynomial
        of the given order.

        :param order:   The order/degree of the polynomial.
        :param x:       An array of point-coordinates. Must have the same number of elements as number of monomials.
        :return:        A square matrix with power-terms of the coordinates of the x-points.
        """
        if order <= 0:
            raise ValueError("Order of polynomial interpolation should be positive")
        dimension = x.shape[1]
        if dimension not in [1, 2, 3]:
            raise ValueError("Unsupported dimension of point data")
        K = Helpers.number_of_monomial_terms(dimension, order)
        if len(x) < K:
            raise ValueError("Need " + str(K) + " x values to interpolate")
        V = np.zeros((K, K), dtype=np.float64)
        if dimension == 1:
            raise ValueError("Dimension one is not yet implemented.")
        elif dimension == 2:
            powers = Helpers.poly2_make_monomial_powers(order)
            # 2022-03-14 Kenny: TODO can probably optimize k-loop away, and rewrite inner part to a
            #                    loop over dimensions. This way the code will be more generic.
            for k in range(K):
                for idx, power in enumerate(powers):
                    V[k, idx] = x[k, 0] ** power[0] * x[k, 1] ** power[1]
        elif dimension == 3:
            raise ValueError("Dimension three is not yet implemented.")
        else:
            raise ValueError("Unsupported number of dimensions.")
        return V


class Poly:
    """
    Base class for all polynomials.
    """

    def __init__(self, dimension: int, order: int) -> None:
        """
        Initialize an instance of Poly2.
        This includes allocating internal arrays for encoding all terms of a two-dimensional polynomial.
        Initializes the shared member variables between all polynomials regardless of their dimension.

        A one dimensional polynomial is given by :math:`P_n(x)`, and a two-dimensional is
        :math:`P_n(x,y)`, and a third dimensional is :math:`P_n(x,y,z)`.

        Here :math:`n` is the order or degree of the polynomial. Polynomials are stored in their canonical form.
        As an example for a two-dimensional polynomial we have

        .. math ::
            P_n(x,y) = a_0 + a_1 x + a_2 y + a_3 x^2 + \cdots + a_K y^n

        The coefficients :math:`a_0, a_1, \cdots, a_K` are stored internally in a coefficients vector. The
        powers of each monomial are stored in a K-by-2 matrix. Such that the :math:`k` canonical term stores the
        powers of :math:`x` and :math:`y` as first and second colum in the   :math:`k`'row.

        .. math ::
            coefficient[k] = a_k,\, powers[k,:] = [i,j]  \quad \Rightarrow \quad a_k x^i y^j

        For a third-dimensional polynomial we simply add a third column to the powers' matrix.

        The data structure layout hence allow us to store polynomial in the same fashion regardless
        of their dimensionality.

        :param dimension:  The dimension of the polynomial. Can be 1, 2 or 3.
        :param order:  The order/degree of the polynomial.

        """
        if dimension not in [1, 2, 3]:
            raise ValueError("Unsupported polynomial dimension")
        self.dimension = dimension
        self.order = order
        self.powers = None
        if dimension == 1:
            raise ValueError("Currently one dimensional polynomial not supported")
        elif dimension == 2:
            self.powers = Helpers.poly2_make_monomial_powers(order)
        elif dimension == 3:
            raise ValueError("Currently third dimensional polynomial not supported")
        else:
            raise ValueError("The dimension of the polynomial must be 1, 2 or 3.")
        self.terms = self.powers.shape[0]
        self.coefficients = np.ones((self.terms,), dtype=np.float64)

    def __str__(self) -> str:
        """
        Convert the Poly instance to a string value.

        :return:  The string value for the polynomial.
        """
        return to_str(self)

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluates the polynomial at the given p-point.

        :param x:  The point where to evaluate the polynomial at x.
        :return:   The value P(x) of the polynomial at position x.
        """
        return evaluate(self, x)


def interpolate(order: int, x: np.ndarray, v: np.ndarray) -> list[Poly]:
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

    :param order:    The order of the resulting polynomials.
    :param x:        An array of point-coordinates.
    :param v:        An matrix of values to be interpolated at the given (x,y) points. Each column's gives one
                     unique polynomial.
    :return:         A list of polynomials that interpolate the given data.
    """
    if order <= 0:
        raise ValueError("Order of polynomial interpolation should be positive")
    dimension = x.shape[1]
    if dimension not in [1, 2, 3]:
        raise ValueError("Unsupported dimension of point data")
    K = Helpers.number_of_monomial_terms(dimension, order)
    if len(x) < K:
        raise ValueError("Need " + str(K) + " points to interpolate")
    if v.shape[0] < K:
        raise ValueError("Need " + str(K) + " values to interpolate")
    V = Helpers.make_vandermonde_matrix(order, x[:K])
    B = np.array(v[:K])
    # 2022-03-14 Kenny: TODO: The condition number of the Vandermonde matrix can be quite large. Hence, this
    #                    approach for solving for an interpolating multivariate polynomial might not always work.
    A = linalg.solve(V, B)
    if v.ndim == 1:
        Ps = [Poly(dimension, order)]
        Ps[0].coefficients = A
    else:
        Ps = [Poly(dimension, order) for _ in range(A.shape[1])]
        for k in range(A.shape[1]):
            Ps[k].coefficients = A[:, k]
    return Ps


def evaluate(P: Poly, x: np.ndarray) -> float:
    """
    Evaluate the given polynomial at the given input point.

    :param P:  The polynomial to be evaluated.
    :param x:  The point x where to evaluate the polynomial.
    :return:   The value of P(x).
    """
    # 2022-03-14 Kenny: TODO this code uses straightforward evaluation of the canonical form. A multivariate
    #                    Horner method might be more appropriate.
    # 2022-03-14 Kenny: TODO this code only works for a single input point. It would be nice to rewrite
    #                     code so x-input could be a whole array of input-points.
    if x.shape[0] != P.dimension:
        raise ValueError("Input point must have same dimension as the polynomial")
    if P.dimension == 1:
        return np.sum(np.multiply(P.coefficients, x[0] ** P.powers[:, 0]))
    elif P.dimension == 2:
        return np.sum(
            np.multiply(
                P.coefficients,
                np.multiply(
                    x[0] ** P.powers[:, 0],
                    x[1] ** P.powers[:, 1]
                )
            )
        )
    elif P.dimension == 3:
        return np.sum(
            np.multiply(
                P.coefficients,
                np.multiply(
                    x[0] ** P.powers[:, 0],
                    np.multiply(
                        x[1] ** P.powers[:, 1],
                        x[2] ** P.powers[:, 2]
                    )
                )
            )
        )
    else:
        raise ValueError("Unsupported dimensional polynomial encountered")


def to_str(P: Poly) -> str:
    """
    Converts the given polynomial into a string value.

    :param P:    The polynomial.
    :return:     A string value that display the polynomial in its canonical form.
    """
    var = ["x", "y", "z"]  # Fixed labels for 1D, 2D and 3D polynomials.
    polynomial = ""
    for idx, power in enumerate(P.powers):
        term = ""
        # First we determine the possible sign to use for next term
        sign = "+"
        if P.coefficients[idx] < 0:
            sign = "-"
        # Second we test if we have a non-zero coefficient. If not we skip this term
        c = math.fabs(P.coefficients[idx])
        if c > 0:
            # Third we decode how the coefficient should be converted to a string.
            if c.is_integer():
                c_str = str(int(c))
            else:
                c_str = str(c)
            # Fourth we decode how the powers of the variables should be converted to a string.
            powers = ""
            for d in range(P.dimension):
                k = power[d]
                if k != 0:
                    if d > 0 and len(powers) > 0:
                        powers += " "
                    if k == 1:
                        powers += var[d]
                    else:
                        powers += var[d] + "^" + str(k)
            # Fifth, we have special rules if coefficient is one and powers are non-zero then we omit writing 1
            #
            if c != 1 and len(powers) > 0:
                term = c_str + " " + powers
            if c == 1 and len(powers) > 0:
                term = powers
            if len(powers) == 0:
                term = c_str
        if len(polynomial) > 0:
            if len(term) > 0:
                polynomial += " " + sign + " " + term
        else:
            polynomial = term
    return polynomial


def derivative(P: Poly) -> list[Poly]:
    """
    This function generates a polynomial that corresponds to the
    derivative of the given polynomial.

    :param P:     The input polynomial.
    :return:       The output polynomials.
    """
    # 2022-03-14 Kenny: TODO: Technically the derivative decreases the order of the polynomial. The current
    #                    implementation correctly computes monomials and their corresponding coefficients, but
    #                    the "powers" pattern will not have the correct "order". We should convert the derivatives
    #                    to a correct one-degree lower polynomial.
    dPs = [Poly(P.dimension, P.order) for _ in range(P.dimension)]
    for d, dP in enumerate(dPs):
        for k, term in enumerate(P.powers):
            dP.powers[k, :] = term
            dP.powers[k, d] = term[d]-1 if term[d] != 0 else 0
            dP.coefficients[k] = term[d]*P.coefficients[k]
    return dPs
