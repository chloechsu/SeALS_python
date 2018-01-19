import numpy as np
import tensorly as tl
from tensors.cp_tensor import CPTensor, get_random_CP_tensor
from exceptions import NumericalError

class CompressionInfo:
    """ Additional info about a compression run.

    Variables
    ------
    ill_conditioned : bool
        whether the ALS matrices were ill-conditioned at any step
    n_iter: int
        number of iterations
    t_step : lsit of floats
        computing time for each iteration during the run
    F_cond: list of floats
        condition numbers of F in each iteration during the run
    errors : list of floats
        errors in each iteration during the run
    iter_with_rank_inc: list of ints
        iterations where the rank of F is increased
    """
    def __init__(self):
        self.ill_conditioned = False
        self.success = False
        self.n_iter = 0
        self.t_step = []
        self.F_cond = []
        self.errors = []
        self.iter_with_rank_inc = []

class Compressor:
    """
    CP tensor compression by ALS.
        
        Given a CP tensor G (represented as a list of factors), 
        try to find a lower rank approximation of G within the desired accuracy.

    Parameters
    ----------
    accuracy : float, required
        desired accuracy
    n_iter_max : int, optional, default is 2000
        max number of iterations
    min_error_dec : float, optional, default is 1e-3
        minimum decrease for error
        (the compressor adds rank when error decrease is below this threshold)
    alpha : float, optional, default is 1e-14
        regularization coefficient
    display_progress : bool, optional, default is false
    verbose : bool, optional, default is false
        whether to print debugging information
    """

    def __init__(self, accuracy, n_iter_max=2000, min_error_dec=1e-3, alpha=1e-14,
            display_progress=False, verbose=False):
        self.accuracy = accuracy
        self.n_iter_max = n_iter_max
        self.min_error_dec = min_error_dec
        self.alpha = alpha
        self.display_progress = display_progress
        self.verbose = verbose

    def get_params(self, **kwargs):
        """ Returns a dictionary of parameters. """
        params = ['n_iter_max', 'min_error_dec', 'alpha', 
                'display_progress', 'verbose']
        return {param_name: getattr(self, param_name) for param_name in params}

    def set_params(self, **params):
        """ Sets the value of the provided parameters. """
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def compress(self, G, initial_guess=None):
        """
        Tries to find a lower rank approximation for G within accuracy.

        Parameters
        ----------
        G : the CPTensor to be approximated
        initial_guess : initial guess of F, default is a random rank-1 tensor
        
        Returns
        -------
        F : CPTensor
            Low rank approximation of G, represented by a list of factors.
        info : CompressionInfo
            Gives additional information about the compression run
        
        """
        assert isinstance(G, CPTensor), "wrong type, not CPTensor"
        G.arrange()

        if initial_guess is None:
            F = get_random_CP_tensor(G.dim, 1)
        else:
            F = initial_guess

        info = CompressionInfo()
        for info.n_iter in xrange(self.n_iter_max):
            # TODO: time the operations
            try:
                F = self.compress_onestep(G, F)
            except NumericalError:
                print "ALS matrix inversion ill-conditioned, returning after \
                        iteration %d" % iteration
                info.ill_conditioned = True
                return F, info
            F.arrange()
            info.F_cond.append(np.linalg.norm(F.lambdas) / F.norm())
            info.errors.append(G.minus(F).norm() / G.norm())

            if self.display_progress:
                print "Iteration %d, error = %2.3f" % (info.n_iter,
                        info.errors[-1])

            if info.errors[-1] <= self.accuracy:
                info.success = True
                return F, info
            
            if info.n_iter > 1 and \
                    np.abs(info.errors[-1] - info.errors[-2]) / info.errors[-2] < \
                    self.min_error_dec:

                if F.rank == G.rank - 1:
                    info.errors[-1] = 0
                    return G, info

                F = F.plus(get_random_CP_tensor(F.dim,1))
                info.iter_with_rank_inc.append(info.n_iter) 
        
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
            M += self.alpha * np.eye(F.rank)
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

