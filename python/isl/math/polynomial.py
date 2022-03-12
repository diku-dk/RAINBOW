import numpy as np
import scipy.linalg as linalg


class Poly1D:

    def __init__(self):
        self.a = None
        self.mono = None


class Poly2D:

    def number_of_monomial_terms(order):
        import math
        return math.comb(order + 2, 2)

    def make_monomial_powers(degree: int):
        """
        :param degree:  The degree of the polynomial
        :return:        A K-by-2 array where each row encodes the powers of a monomial and K would be the total number of monomials.
        """
        patterns = []
        if degree < 0:
            raise ValueError("Polynomial degree must be non-negative")
        for p in range(degree + 1):
            for j in range(p + 1):
                i = p - j
                patterns.append([i, j])
        powers = np.array(patterns, dtype=int)
        return powers

    def make_vandermonde_matrix(P, order: int):
        """

        :param P:
        :param order:
        :return:
        """
        K = number_of_monomial_terms(order)
        V = np.zeros((K, K), dtype=np.float64)
        powers = make_monomial_powers(order)
        for k in range(K):
            for idx, power in enumerate(powers):
                V[k, idx] = P[k, 0]**power[0] * P[k, 1]**power[1]
        return V

    def compute_polynomial_coefficients(V, b):
        a = linalg.solve(V, b)
        return a

    def make_poly_str(degree: int):
        """
        :param degree:
        :return:
        """
        polynomial = ""
        powers = make_monomial_powers(degree)
        for idx, term in enumerate(powers):
            mononial = ""
            if term[0] == 0 and term[1] == 0:
                mononial += "1"
            else:
                if term[0] > 0:
                    mononial += "x^" + str(term[0])
                if term[1] > 0:
                    mononial += "y^" + str(term[1])
            if idx > 0:
                polynomial += " + " + mononial
            else:
                polynomial = mononial
        return polynomial


class Poly3D:

    def __init__(self):
        self.a = None
        self.mono = None









"""
We want to find coefficients of an interpolating polynomial 

    P_n(x,y) = sum_{\forall 0 \leq i,j \leq n, i+j=n} a_{i,j} x^i y^j
    
Let K be the number of terms in the polynomial P_n then we have
K-coefficients we want to determine.

The polynomial can be written in matrix vector notation as follows

    P_n(x,y) = [1 x y x^2 y^2 x^3 x^2 y  y^3 ..... ]  [a_0 a_1 ... a_{K-1}]^T
    
The idea is that we have K-points in the plane and for each point we know the value that we wish to
interpolate. Hence, we know

    P_n(x_i, y_i) = z_i for i=0 to K-1

This allows us to form a linear system of K equations.  The coefficient matrix, V, is known
as the generalized Vandermonde matrix, and has this form,

V = [
[1 x_0 y_0 x_0^2 y_0^2 x_0^3 x_0^2 y_0  y_0^3 ..... ]
[1 x_1 y_1 x_1^2 y_1^2 x_1^3 x_1^2 y_1  y_1^3 ..... ]
...
[1_{K-1} x_{K-1} y_{K-1} x_{K-1}^2 y_{K-1}^2 x_{K-1}^3 x_{K-1}^2 y_{K-1}  y_{K-1}^3 ..... ]
]

The values z_i forms the right-hand-side vector, 

b = [z_0 z_1 ... z_{K-1}]^^

and now we can solve the system V a = z for the coefficient vector a. 
"""




if __name__ == "__main__":
    for p in range(5):
        print("Polynomial of order ", p, " has ", number_of_monomial_terms(p), "monomials terms and looks like this ",
              make_poly_str(p))
    points = np.random.rand(100, 2)*100
    V = make_matrix(points, 4)
    B = np.eye(V.shape[0])
    A = compute_polynomial_coefficients(V, B)
    print('Generalized Vandermonde matrix: ', V)
    print('Polynomial coefficients: ', A)
