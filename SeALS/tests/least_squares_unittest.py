import unittest
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensors.cp_tensor import CPTensor, CPTensorOperator
from tensors.cp_tensor import get_random_CP_tensor, get_random_CP_tensor_operator
from tensors.cp_tensor import get_identity_tensor_operator 
from core.least_squares import Solver, error
from core.info import ALSOptions
from core.exceptions import NumericalError
from copy import deepcopy

class TestLeastSquaresSolver(unittest.TestCase):

    def assertArrayAlmostEqual(self, A, B, eps=1e-7):
        self.assertTrue(np.max(np.abs(A-B)) < eps)

    def get_random_dim_and_rank(self):
        while True:
            # the tensorly package works only when ndim >= 2
            ndim = np.random.randint(2, 5)
            dim = [np.random.randint(1, 5) for i in xrange(ndim)]
            rank = np.random.randint(1, 5)
            if np.prod(np.array(dim)) > rank:
                return dim, rank

    def test_random_A(self, repeats=100, accuracy=1e-3, use_SeALS=False,
            with_initial_F=False):
        success_rate = 0.0
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            F_true = get_random_CP_tensor(dim, rank)
            A = get_random_CP_tensor_operator([(d,d) for d in dim], rank)
            G = A.multiply(F_true)
            # set max rank to be true rank + 1 (if set to true rank, the success
            # rate is lower at about 0.7)
            solver = Solver(ALSOptions(accuracy=accuracy, 
                tol_error_dec=1e-1, error_type='total', max_rank=F_true.rank+1),
                use_SeALS=use_SeALS)
            if with_initial_F:
                F_sol, info = solver.solve(A, G, F_true) 
            else:
                F_sol, info = solver.solve(A, G)
            if info.success:
                self.assertLessEqual(A.multiply(F_sol).minus(G).norm(), accuracy)
                self.assertLessEqual(F_sol.rank, F_true.rank+1)
                success_rate += 1.0 / repeats

        if with_initial_F:
            self.assertGreaterEqual(success_rate, 0.99)
        else:
            self.assertGreaterEqual(success_rate, 0.8)

    def test_random_A_with_initial_guess(self, repeats=100, accuracy=1e-5):
        self.test_random_A(repeats, accuracy, with_initial_F=True)
    
    def test_identity_A(self, repeats=100, accuracy=1e-3, use_SeALS=False):
        """ Compression is a special case when A is the identity.
            We use the same test as for compression.
        """
        success_rate = 0.0
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            G_plus_G = G.plus(G)
            A = get_identity_tensor_operator(G_plus_G.dim)
            # set max rank to be true rank + 1 (if set to true rank, the success
            # rate is lower at about 0.6)
            solver = Solver(ALSOptions(accuracy=accuracy, tol_error_dec=1e-1,
                max_rank=G.rank+1, error_type='total'), use_SeALS=use_SeALS)
            F, info = solver.solve(A, G_plus_G)
            if info.ill_conditioned:
                # retry if the matrix becomes ill-conditioned during ALS
                t -= 1
                continue
            if info.success:
                self.assertLessEqual(F.rank, G.rank+1)
                self.assertLessEqual(F.minus(G_plus_G).norm(), accuracy) 
                success_rate += 1.0 / repeats
        self.assertGreater(success_rate, 0.8)

    def test_SeALS_random_A(self, repeats=100, accuracy=1e-3):
        self.test_random_A(repeats, accuracy, use_SeALS=True)

    def test_SeALS_random_A_with_initial_guess(self, repeats=100, 
            accuracy=1e-5):
        self.test_random_A(repeats, accuracy, with_initial_F=True, 
                use_SeALS=True)
    
    def test_SeALS_identity_A(self, repeats=100, accuracy=1e-3):
        self.test_identity_A(repeats, accuracy, use_SeALS=True)
