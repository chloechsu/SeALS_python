import numpy as np
from copy import deepcopy
import time
import tensorly as tl
from tensors.cp_tensor import CPTensor, CPTensorOperator, get_random_CP_tensor
from exceptions import NumericalError
from info import SolverInfo, ALSOptions

def error(A, F, G, error_type):
    """
    Helper function to compute the error in AF = G.

    Parameters
    ----------
    A, F, G : CPTensor
    error_type : 'average' or 'total'
    """
    if error_type == 'average':
        normalization = np.sqrt(np.prod(G.dim))
    else:
        normalization = 1.0
    return A.multiply(F).minus(G).norm() / normalization

class Solver:
    """
    Solve AF = G for CP tensors by ALS or SeALS.
    The ALS algorithm is described in Beylkin and Mohlenkamp 2005.
    The SeALS algorithm is described in Stefansson and Leong 2016.
    
    Parameters
    ----------
    options : an instance of ALSOptions
    use_SeALS : whether use the unmodified ALS algorithm or SeALS
    """

    def __init__(self, options, use_SeALS=False):
        assert isinstance(options, ALSOptions), "must intialize with ALSOptions"
        self.options = options
        self.use_SeALS = use_SeALS

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
        info.errors.append(error(A, F, G, self.options.error_type))
        info.F_cond.append(np.linalg.norm(F.lambdas) / F.norm())

        AA, AG = self.compute_inner_products(A, G)

        # initialize the number of oscilizations (increasing error) for SeALS
        n_osc = 0
        # initialize saved F for SeALS
        F_saved = CPTensor([])

        # In each iteration, do the following:
        # 1. compute current error and log
        # 2. if the error is within target accuracy, return
        # 3. if error decreases less than tol_error_dec, increase the rank of F
        #   3.1 if F is already at max_rank, can't increase so return
        #   3.2 to increase the rank of F, add a new preconditioned rank1 tensor
        # 4. do one iteration of alternating gradient descents 
        #    (by calling ALS_least_squares_onestep)
        # 5.
        #   5A (ALS without the SeALS variant)
        #       if the matrix is ill conditioned in 4, return previous F
        #   5B (SeALS)
        #       if the matrix is ill conditioned in 4, or if n_osc reaches
        #       tol_osc, save F and restart ALS with a new preconditoned rank 1
        #       tensor as F
        for info.n_iter in xrange(self.options.n_iter_max):

            if info.errors[-1] <= self.options.accuracy:
                info.success = True
                return F.plus(F_saved), info

            if self.options.verbose:
                print "error = %2.9f, rank(F) = %d, starting iteration %d.." % \
                (info.errors[-1], F.rank, info.n_iter)

            F.arrange()
            
            flag_SeALS_restart = False

            try:
                start_time = time.clock()
                F = self.ALS_least_squares_onestep(A, G, F, AA, AG)
                info.t_step.append(time.clock() - start_time)
            except NumericalError:
                if not self.use_SeALS:
                    print "ill-conditioned after iteration %d" % info.n_iter
                    info.ill_conditioned = True
                    return F, info
                flag_SeALS_restart = True

            info.F_cond.append(np.linalg.norm(F.lambdas) / F.norm())
            info.errors.append(error(A, F, G, self.options.error_type))

            if info.errors[-1] > info.errors[-2]:
                n_osc = n_osc + 1
                if self.use_SeALS and n_osc > self.options.tol_osc: 
                    flag_SeALS_restart = True

            # if error decreases less than tol_error_dec,
            # Increase rank of F by adding a preconditioned rank 1 tensor
            res_dec = np.abs(info.errors[-2] - info.errors[-1]) / info.errors[-2]
            if res_dec < self.options.tol_error_dec:
                # check we are not exceeding max rank
                if F_saved.rank + F.rank >= self.options.max_rank:
                    if len(info.restarts) > 0:
                        print info.restarts
                    print "achieved max rank after iteration %d" % info.n_iter
                    return F.plus(F_saved), info
                if self.use_SeALS and F.rank >= self.options.tol_rank_restart:
                    flag_SeALS_restart = True
                    
                # add a precondtioned rank 1 tensor to F
                if not flag_SeALS_restart:
                    info.iter_with_rank_inc.append(info.n_iter) 
                    if self.options.verbose:
                        print "Increasing rank of F at iteration %d" % info.n_iter

                    F = F.plus(self.get_preconditioned_rank1_tensor(A,
                        G.minus(A.multiply(F))))
                    # TODO: implement the python equivalent of fixsigns in MATLAB??

            # if the ALS matrix inversion is ill conditioned or if n_osc exceeds
            # tol_osc, restart ALS with a new preconditioned rank 1 tensor
            if self.use_SeALS and flag_SeALS_restart: 
                info.restarts.append(info.n_iter) # logging
                n_osc = 0 # reset n_osc
                F_saved = F.plus(F_saved)
                # if the smallest normalization constant divided by the
                # largest normalization constant is small enough, return
                # TODO: I just copied this criterion from MATLAB but I don't
                # really understand why the threshold is the sqrt of accuracy
                if F_saved.rank >= self.options.max_rank or \
                        F_saved.getNormalizationRatio() < \
                        np.sqrt(self.options.accuracy):
                    if len(info.restarts) > 0:
                        print info.restarts
                    print "achieved max rank after iteration %d" % info.n_iter
                    return F_saved, info
                # update G
                G = G.minus(A.multiply(F))
                F = self.get_preconditioned_rank1_tensor(A, G)
                AA, AG = self.compute_inner_products(A, G)
        
            # update the current error since we might have changed F
            info.errors[-1] = error(A, F, G, self.options.error_type)

        if len(info.restarts) > 0:
            print info.restarts
        return F.plus(F_saved), info
    
    def ALS_least_squares_onestep(self, A, G, F, AA, AG):
        """ A single iteration of the ALS least squares algorithm.
            
            parameters
            ----------
            AA : list of 2darrays
                precomputed matrix A_d^T A_d for each dimension 
                A_d^T A_d is size (A.dim[d][1]*A.rank) x (A.dim[d][1]*A.rank)
            AG : list of 2darrays
                precomputed A_d^T G_d for each dimension
                A_d^T G_d is size (A.dim[d][1]) x (A.rank * G.rank)
            F : CPTensor

            returns
            -------
            F : updated CPTensor

        """
        assert not np.any(np.isnan(F.lambdas))
        F.distribute_lambda()
        
        for k in xrange(len(F.dim)):
            # calculate B matrix
            B = np.zeros((F.rank*F.dim[k], F.rank*F.dim[k]))
            for i, j in np.ndindex(F.rank, F.rank):
                Mt = np.zeros((F.dim[k], F.dim[k]))
                for ia, ja in np.ndindex(A.rank, A.rank):
                    FAAF = 1.0;
                    for d in xrange(len(F.dim)):
                        if d == k:
                            continue
                        A_ia_ja = AA[d][ia*F.dim[d]:(ia+1)*F.dim[d],
                                ja*F.dim[d]:(ja+1)*F.dim[d]]
                        FAAF *= np.dot(F.factors[d][:,i].transpose(),
                                np.dot(A_ia_ja, F.factors[d][:,j]))
                    Mt += FAAF * AA[k][ia*F.dim[k]:(ia+1)*F.dim[k],
                            ja*F.dim[k]:(ja+1)*F.dim[k]]
                if i == j:
                    B[i*F.dim[k]:(i+1)*F.dim[k], j*F.dim[k]:(j+1)*F.dim[k]] = \
                            Mt + self.options.alpha * np.eye(F.dim[k])
                else:
                    B[i*F.dim[k]:(i+1)*F.dim[k], j*F.dim[k]:(j+1)*F.dim[k]] = Mt
            
            # calculate b vector
            b = np.zeros(F.rank*F.dim[k])
            for i in xrange(F.rank):
                GAF = np.ones((1, A.rank*G.rank))
                for d in xrange(len(A.dim)):
                    if d == k:
                        continue
                    GAF = GAF * np.dot(F.factors[d][:,i].reshape(1,-1), AG[d])
                b[i*F.dim[k]:(i+1)*F.dim[k]] = np.dot(AG[k],
                        GAF.reshape(-1,1)).reshape(F.dim[k])

            # TODO: record B and b for debugging
            
            if np.linalg.cond(B) > 1.0 / np.finfo(float).eps:
                raise NumericalError("B is ill-conditioned")
            try:
                u = np.linalg.solve(B, b)
            except np.linalg.LinAlgError:
                raise NumericalError("B is ill-conditioned")
            
            F.factors[k] = u.reshape(F.rank, F.dim[k]).transpose()

        F.arrange()
        # TODO: fixsigns???
        return F

    def compute_inner_products(self, A, G):
        A.tensor.arrange()
        A.tensor.distribute_lambda()
        G.arrange()
        G.distribute_lambda()
        # precompute some inner products
        AA = []
        AG = []
        for d in xrange(len(A.dim)):
            G_d = G.factors[d]
            # The factor is of dimension (A.dim[d][0]*A.dim[d][1]) x A.rank
            A_d = A.tensor.factors[d].reshape(A.dim[d][0], A.dim[d][1]*A.rank)
            AG.append(np.dot(A_d.transpose(),
                G_d).reshape(A.dim[d][1],A.rank*G.rank))
            # swap indices for easier array slicing in each iteration
            A_d = np.swapaxes(A_d.reshape(A.dim[d][0],A.dim[d][1],A.rank), 1,
                    2).reshape(A.dim[d][0], A.rank * A.dim[d][1])
            AA.append(np.dot(A_d.transpose(), A_d))
        return AA, AG
    
    def get_preconditioned_rank1_tensor(self, A_pre, G_pre):
        """ Preconditions the new rank-1 tensor to be added to F.
        """
        F = get_random_CP_tensor([d[1] for d in A_pre.dim], 1)
        AA_pre, AG_pre = self.compute_inner_products(A_pre, G_pre)
        error_prev = error(A_pre, F, G_pre, self.options.error_type)

        for iter_pre in xrange(self.options.n_iter_max_precond):
            F = self.ALS_least_squares_onestep(A_pre, G_pre, F, AA_pre, AG_pre)

            error_curr = error(A_pre, F, G_pre, self.options.error_type)
            if self.options.verbose:
                print "Preconditioning new rank1 tensor: iteration %d, error = %2.9f" % \
                (iter_pre, error_curr)
            if error_curr == 0 or (error_prev - error_curr) / error_prev <= \
                    self.options.tol_error_dec_precond:
                return F
            error_prev = error_curr
            
        return F
