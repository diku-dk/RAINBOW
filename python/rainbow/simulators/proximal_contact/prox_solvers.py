from rainbow.simulators.proximal_contact.proximal_operators import *
from rainbow.simulators.proximal_contact.gauss_seidel_solver import GaussSeidelSolver, ParallelGaussSeidelSolver
from rainbow.simulators.proximal_contact.jacobi_solver import ParallelJacobiSolver, ParallelJacobiHybridSolver


def solve(J, WJT, b, mu, friction_solver, engine, stats, debug_on, prefix="", scheme="gauss_seidel"):
    """_summary_

    Args:
         J (ArrayLike): The contact jacobi matrix.
        WJT (ArrayLike): The WJ^T matrix, here W = M^{-1}.
        b (ArrayLike): b = Ju^t + ∆t JM^{-1}h + EJu^t
        mu (float): The coefficient of friction.
        friction_solver (callable): The proximal operator of friction cone function.
        engine (Object): The engine object.
        stats (dict): The statistics information.
        debug_on (boolean): Whether to debug.
        prefix (string): The prefix of the statistics information.
        scheme (str, optional): The scheme of proximal solver. Defaults to "gauss_seidel", you can use one of 'gauss_seidel', 'parallel_gauss_seidel', 'parallel_jacobi', 'parallel_jacboi_hybrid'.

    Raises:
        NotImplementedError: Unknown solver scheme.

    Returns: The new contact force and the statistics information.
    """

    # If the prefix is empty, use the scheme as the prefix
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