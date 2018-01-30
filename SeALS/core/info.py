class ALSOptions:
    """ A class to store ALS options for solver.
    
    Parameters
    ----------
    For both least squares solver and compressor:

    accuracy : float, required
        desired accuracy
    n_iter_max : int, optional, default is 2000
        max number of iterations
    max_rank : int, optional, default is 30
        max tolerated rank for F
    tol_error_dec : float, optional, default is 1e-3
        minimum decrease for error
        (the compressor adds rank when error decrease is below this threshold)
    alpha : float, optional, default is 1e-12
        regularization coefficient
    verbose : bool, optional, default is false
        whether to display progress 
    debugging : bool, optional, default is false
        whether to store additional debugging info

    Only for least squares solver:

    error_type : 'average' or 'total', optional, default is point average
    tol_error_dec_precond : float, optional, default is 1e-2
        tolerated decrease for precondition
    n_iter_max_precond : int, optional, default is 15
        maximum number of tolerated iterations for preconodition

    """

    def __init__(self, accuracy, n_iter_max=2000, 
            max_rank=30, tol_error_dec=1e-3, alpha=1e-12,
            error_type='average', 
            tol_error_dec_precond=1e-2, n_iter_max_precond=15,
            verbose=False, debugging=False):
        self.accuracy = accuracy
        self.n_iter_max = n_iter_max
        self.tol_error_dec = tol_error_dec
        self.alpha = alpha
        assert error_type == 'average' or error_type == 'total', \
                "unknown error type"
        self.error_type = error_type
        self.tol_error_dec_precond = tol_error_dec_precond
        self.n_iter_max_precond = n_iter_max_precond
        self.verbose = verbose
        self.debugging = debugging

    def get_params(self, **kwargs):
        """ Returns a dictionary of parameters. """
        params = ['n_iter_max', 'tol_error_dec', 'alpha', 
                'display_progress', 'verbose', 'debugging']
        return {param_name: getattr(self, param_name) for param_name in params}

    def set_params(self, **params):
        """ Sets the value of the provided parameters. """
        for param, value in params.items():
            setattr(self, param, value)
        return self


class SolverInfo:
    """ A class to store additional info from a solver.

    Variables
    ------
    For both least squares solver and compressor:

    ill_conditioned : bool
        whether the ALS matrices were ill-conditioned at any step
    n_iter: int
        number of iterations
    t_step : lsit of floats
        computing time (in seconds) for each iteration during the run
    F_cond: list of floats
        condition numbers of F in each iteration during the run
    errors : list of floats
        errors in each iteration during the run
    iter_with_rank_inc: list of ints
        iterations where the rank of F is increased
    
    Only for least squares solver when debugging is on:

    F_record : list of F in each iteration
    B_record : list of ALS matrix G in each iteration
    b_record : list of RHS vector b in each iteration

    """
    def __init__(self):
        """ For both compression and least squares solver:
        """
        self.ill_conditioned = False
        self.success = False
        self.n_iter = 0
        # t_step starts from iteration 1
        self.t_step = [None]
        # self.F_cond[0] is the condition number for initial guess
        self.F_cond = []
        # self.errors[0] is the error for initial guess
        self.errors = []
        self.iter_with_rank_inc = []

        """ Only for least squares solver:
        """
        # all F in each iteration
        self.F_record = []
        # all ALS matrix B in each iteration 
        self.B_record = []
        # all RHS vector b in each iteration
        self.b_record = []
