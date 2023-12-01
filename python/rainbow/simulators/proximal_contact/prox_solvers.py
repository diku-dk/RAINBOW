from rainbow.simulators.proximal_contact.proximal_operators import *
from rainbow.simulators.proximal_contact.gauss_seidel_solver import GaussSeidelSolver, ParallelGaussSeidelSolver
from rainbow.simulators.proximal_contact.jacobi_solver import ParallelJacobiSolver, ParallelJacobiHybridSolver


def solve(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix="", scheme="gauss_seidel"):
    """ This is a function to collect all schemes of the proximal solvers.

    :param J: The contact jacobi matrix.
    :param WJT: The WJ^T matrix, here W = M^{-1}.
    :param b: b = Ju^t + ∆t JM^{-1}h + EJu^t
    :param mu: The coefficient of friction.
    :param friction_solver: The proximal operator of friction cone function.
    :param engine: The engine object.
    :param stats: The statistics information.
    :param debug_on:  Whether to debug.
    :param prefix: The prefix of the statistics information., defaults to ""
    :param scheme: The scheme of proximal solver. Defaults to "gauss_seidel", you can use one of 'gauss_seidel', 'parallel_gauss_seidel', 'parallel_jacobi', 'parallel_jacboi_hybrid'. defaults to "gauss_seidel"
    :raises NotImplementedError: Unknown solver scheme.
    :return: The new contact force and the statistics information.
    """

    # If the prefix is empty, use the scheme name as the prefix
    prefix = prefix if prefix != "" else scheme + "_"

    if scheme == "gauss_seidel":
        solver = GaussSeidelSolver(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix)
    elif scheme == "parallel_gauss_seidel":
        solver = ParallelGaussSeidelSolver(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix)
    elif scheme == "parallel_jacobi":
        solver = ParallelJacobiSolver(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix)
    elif scheme == "parallel_jacboi_hybrid":
        solver = ParallelJacobiHybridSolver(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix)
    else:
        raise NotImplementedError("Unknown solver scheme: {}, you can use one of 'gauss_seidel', 'parallel_gauss_seidel', 'parallel_jacobi', 'parallel_jacboi_hybrid'.".format(scheme))
    
    return solver.solve()
