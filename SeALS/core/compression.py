import numpy as np
import tensorly as tl
from ..tensors.cp_tensor import CPTensor
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
    error : list of floats
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
        self.error = []
        self.iter_with_rank_inc = []

class Compressor:
    """
    CP tensor compression by ALS.
        
        Given a CP tensor G (represented as a list of factors), 
        try to find a lower rank approximation of G within the desired accuracy.

    Parameters
    ----------
    acc : float, required
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

    def __init__(self, acc, n_iter_max=2000, min_error_dec=1e-3, alpha=1e-14,
            display_progress=False, verbose=False):
        self.acc = acc
        self.n_iter_max = n_iter_max
        self.min_error_dec = min_error_dec
        self.alpha = alpha
        self.display_progress = display_progress
        self.verbose = verbose
        self.G = None

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

    def set_tensor(self, G):
        assert type(G) is CPTensor, "wrong type, not CPTensor"
        self.G = G
        self.G.arrange()

    def compress(initial_guess=None):
        """
        Tries to find a lower rank approximation for G within accuracy.

        Parameters
        ----------
        initial_guess : initial guess of F, default is a random rank-1 tensor
        
        Returns
        -------
        F : CPTensor
            Low rank approximation of G, represented by a list of factors.
        info : CompressionInfo
            Gives additional information about the compression run
        
        """
        if initial_guess is None:
            F = get_random_CP_tensor(G.dim, 1)
        else:
            F = initial_guess

        info = CompressionInfo()
        for info.n_iter in xrange(n_iter_max):
            # TODO: time the operations
            try:
                F = compress_onestep(F)
            except NumericalError:
                print "ALS matrix inversion ill-conditioned, returning after \
                        iteration %d" % iteration
                info.ill_conditioned = True
                return F, info
            
            F.arrange()
            info.F_cond.append(np.linalg.norm(F.lambdas) / F.norm())
            info.errors.append(self.G.minus(F).norm() / self.G.norm())

            if display_progress:
                print "Iteration %d, error = %2.3f" % (info.n_iter, error)

            if info.errors[-1] <= acc:
                info.success = True
                return F, info
            
            if np.abs(info.errors[-1] - info.errors[-2]) / info.errors[-2] < \
                    self.min_error_dec:

                if F.rank == self.G.rank - 1:
                    info.errors[-1] = 0
                    return self.G, info

                F = F.add(get_random_CP_tensor(F.dim,1))
                info.iter_with_rank_inc.append(iteration) 
        
        return G, info
