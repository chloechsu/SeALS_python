import unittest
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from tensors.cp_tensor import get_random_CP_tensor, CPTensor
from core.compression import Compressor
from core.info import ALSOptions
from core.exceptions import NumericalError

class TestCompression(unittest.TestCase):

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
    
    def test_compress_G_plus_G(self, repeats=20, accuracy=1e-3):
        success_rate = 0.0
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            G_plus_G = G.plus(G)
            opts = ALSOptions(accuracy=accuracy, tol_error_dec=1e-2)
            c = Compressor(opts)
            F, info = c.compress(G_plus_G)
            if info.ill_conditioned:
                # retry if the matrix becomes ill-conditioned during ALS
                t -= 1
                continue
            self.assertLessEqual(F.minus(G_plus_G).norm() / G_plus_G.norm(), 
                    accuracy)
            self.assertLessEqual(F.rank, 2 * G.rank)
            if F.rank == G.rank:
                success_rate += 1.0 / repeats
        self.assertGreater(success_rate, 0.5)


    def test_compress_with_initial_guess(self, repeats=10, accuracy=1e-7):
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            G_plus_G = G.plus(G)
            opts = ALSOptions(accuracy=accuracy)
            c = Compressor(opts)
            F, info = c.compress(G_plus_G, CPTensor(G.factors, 2*G.lambdas))
            self.assertLessEqual(F.minus(G_plus_G).norm() / G_plus_G.norm(), 
                    accuracy)
            self.assertLessEqual(F.rank, G.rank)
