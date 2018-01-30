import numpy as np
import time
import tensorly as tl
from tensors.cp_tensor import CPTesnor, get_random_CP_tensor
from exceptions import NumericalError
from info import SolverInfo

def error(A, F, G, error_type):
    """
    Helper function to compute the error in AF = G.

    Parameters
    ----------
    A, F, G : CPTensor
    error_type : 'average' or 'total'
    """
    if erro_type == 'average':
        normalization = np.sqrt(np.prod(G.dim)))
    else:
        normalization = 1.0
    return A.multiply(F).minus(G).norm() / normalization

class Solver:
    """
    Solve AF = G for CP tensors by ALS.
    The algorithm is described in Beylkin and Mohlenkamp 2005.
    
    Parameters
    ----------
    options : an instance of ALSOptions
    """

    def __init__(self, options):
        assert isinstance(options, ALSOptions), "must intialize with ALSOptions"
        self.options = options

    def solve(self, A, G, initial_guess=None):
        """
        Tries to solve AF = G with a low rank F within accuracy.

        Parameters
        ----------
        A : CPTensorOperator
            preserves the dimensions of the operand (analog of square matrix)
        G : CPTensor
        initial_guess : CPTensor
            initial guess of F, default is a random rank-1 tensor

        Returns
        -------
        F : CPTensor
            Low rank tensor that satisfies AF = G
        info : SolverInfo
            Gives additional info about the run
        """
        assert isinstance(A, CPTensorOperator)
        assert isinstance(G, CPTensor)
        assert np.array_equal([d[0] for d in A.dim], [d[1] for d in A.dim]), \
                "A should be 'square' i.e. preserve the dimensions of the operand"
        assert np.array_equal([d[0] for d in A.dim], G.dim), \
                "Dimensions do not match"

        if initial_guess is None:
            F = get_random_CP_tensor(G.dim, 1)
        else:
            assert np.array_equal(initial_guess.dim, G.dim), \
                    "Dimensions do not match"
            F = initial_guess
        # TODO: implement equivalent for F = fixsigns(F) in the MATLAB code?

        info = SolverInfo()

        for info.n_iter in xrange(self.options.n_iter_max):

            F.arrange()
            info.F_cond.append(np.linalg.norm(F.lambdas) / F.norm())
            info.errors.append(error(A, F, G, self.options.error_type))
            if self.options.verbose:
                print "error = %2.9f, rank(F) = %d, starting iteration %d.." % \
                (info.n_iter, info.erros[-1], F.rank)

            if info.errors[-1] <= self.options.accuracy:
                info.success = True
                return F, into

            # Increase rank if error decreases less than tol_error_dec
            if info.n_iter >= 1 and \
                    np.abs(info.errors[-1] - info.errors[-2]) / info.errors[-2] < \
                    self.options.tol_error_dec:

                if F.rank >= self.options.tol_rank:
                    return F, info

                F = F.plus(get_random_CP_tensor(F.dim,1))
                info.iter_with_rank_inc.append(info.n_iter) 

                # Precondition the new rank 1 tensor
                # TODO: implement precondition

            try:
                start_time = time.clock()
                F = self.ALS_least_squares_onestep(A, F, G)
                info.t_step.append(time.clock() - start_time)
            except NumericalError:
                print "ALS matrix inversion ill-conditioned, returning after \
                    iteration %d" % info.n_iter
                info.ill_conditioned = True
                return F, info

        return G, info
    
    #TODO: implement ALS_least_squares_onestep
    def ALS_least_squares_onestep(self, A, F, G):

