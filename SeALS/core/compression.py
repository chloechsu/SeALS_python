import numpy as np
import time
import tensorly as tl
from tensors.cp_tensor import CPTensor, get_random_CP_tensor
from exceptions import NumericalError
from info import SolverInfo, ALSOptions

class Compressor:
    """
    CP tensor compression by ALS.
        
        Given a CP tensor G (represented as a list of factors), 
        try to find a lower rank approximation of G within the desired accuracy.

    Parameters
    ----------
    options : an instance of ALSOptions 
    """

    def __init__(self, options):
        assert isinstance(options, ALSOptions), "must intialize with ALSOptions"
        self.options = options

    def compress(self, G, initial_guess=None):
        """
        Tries to find a lower rank approximation for G within accuracy.

        Parameters
        ----------
        G : CPTensor
            The CPTensor to be approximated
        initial_guess : CPTensor
            Initial guess of F, default is a random rank-1 tensor
        
        Returns
        -------
        F : CPTensor
            Low rank approximation of G
        info : SolverInfo
            Gives additional information about the compression run
        
        """
        assert isinstance(G, CPTensor), "wrong type, not CPTensor"
        G.arrange()

        if initial_guess is None:
            F = get_random_CP_tensor(G.dim, 1)
        else:
            F = initial_guess
            assert np.array_equal(F.dim, G.dim), \
                    "Dimensions do not match"

        info = SolverInfo()

        for info.n_iter in xrange(self.options.n_iter_max):

            F.arrange()
            info.F_cond.append(np.linalg.norm(F.lambdas) / F.norm())
            info.errors.append(G.minus(F).norm() / G.norm())
            if self.options.verbose:
                print "error = %2.3f, starting iteration %d.." % (info.n_iter,
                        info.errors[-1])
            if info.errors[-1] <= self.options.accuracy:
                info.success = True
                return F, info

            # Increase rank if error decreases less than tol_error_dec
            if info.n_iter >= 1 and \
                    np.abs(info.errors[-1] - info.errors[-2]) / info.errors[-2] < \
                    self.options.tol_error_dec:

                if F.rank == G.rank - 1:
                    info.errors[-1] = 0
                    return G, info

                F = F.plus(get_random_CP_tensor(F.dim,1))
                info.iter_with_rank_inc.append(info.n_iter) 
        
            try:
                start_time = time.clock()
                F = self.compress_onestep(G, F)
                info.t_step.append(time.clock() - start_time)
            except NumericalError:
                print "ALS matrix inversion ill-conditioned, returning after \
                        iteration %d" % info.n_iter
                info.ill_conditioned = True
                return G, info

        return G, info

    def compress_onestep(self, G, F):
        """ One step of the ALS algorithm (iterates through all dimensions)
        """

        # FF[m, i, j] = <F_m^i, F_m^j>
        FF = np.zeros((len(F.dim), F.rank, F.rank))
        for m in xrange(len(F.dim)):
            FF[m, :, :] = np.dot(F.factors[m].transpose(), F.factors[m])
        # GF[m, i, j] = <G_m^i, F_m^j>
        GF = np.zeros((len(F.dim), G.rank, F.rank))
        for m in xrange(len(F.dim)):
            GF[m, :, :] = np.dot(G.factors[m].transpose(), F.factors[m])

        for k in xrange(len(F.dim)):
            idx = range(0, k) + range(k+1, len(F.dim)) 
            # M[i,j] = Prod_{m != k} <F_m^i, F_m^j>
            # M has shape F.rank x F.rank
            M = np.prod(FF[idx,:,:], axis=0)
            # M += alpha * Identity for regularization
            M += self.options.alpha * np.eye(F.rank)
            # multiply lambda into kth factor
            Gk = G.lambdas.reshape(1,-1) * G.factors[k]
            # N[i] = Sum_{j=1...G.rank} G_k^j Prod_{m != k} <F_m^i, G_m^j>
            # N has shape F.rank x M_k
            N = np.dot(Gk, np.prod(GF[idx,:,:], axis=0)).transpose()

            if np.linalg.cond(M) > 1e13:
                raise NumericalError("matrix ill-conditioned")

            # normalize each column of Fk
            Fk = np.linalg.solve(M, N).transpose()
            # Fk has shape M_k x F.rank
            Fk_norm = np.linalg.norm(Fk, axis=0)
            F.factors[k] = Fk / Fk_norm.reshape(1,-1)
            F.lambdas = Fk_norm

            # update inner product in FF and GF
            FF[k,:,:] = np.dot(F.factors[k].transpose(), F.factors[k])
            GF[k,:,:] = np.dot(G.factors[k].transpose(), F.factors[k])
        
        return F

