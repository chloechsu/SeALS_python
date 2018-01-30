import unittest
from tensors.cp_tensor import get_random_CP_tensor, CPTensor, CPTensorOperator
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

class TestCPTensorMethods(unittest.TestCase):

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

    def test_get_random_CP_tensor(self, repeats=10):
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            self.assertEqual(G.rank, rank)
            self.assertEqual(G.dim, dim)

    def test_get_full_tensor(self, repeats=10):
        for t in xrange(repeats): 
            dim, rank = self.get_random_dim_and_rank()
            X = tl.tensor(np.random.random(dim))
            try:
                factors = parafac(X, rank=rank)
            except np.linalg.LinAlgError:
                # if the generated tensor is singular, retry
                t -= 1
                continue
            G = CPTensor(factors)
            self.assertArrayAlmostEqual(G.get_full_tensor(),
                    tl.kruskal_to_tensor(factors))

    def test_norm(self, repeats=10):
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            self.assertAlmostEqual(G.norm(), 
                    np.linalg.norm(G.get_full_tensor()))

    def test_normalize(self, repeats=10):
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            G_copy = np.copy(G.get_full_tensor())
            G.normalize()
            for f in G.factors:
                for r in xrange(G.rank):
                    self.assertAlmostEqual(np.linalg.norm(f[:,r]), 1)
            self.assertArrayAlmostEqual(G_copy, G.get_full_tensor())

    def test_arrange(self, repeats=10):
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            G_copy = np.copy(G.get_full_tensor())
            G.arrange()
            self.assertTrue(np.array_equal(np.argsort(G.lambdas), 
                np.arange(G.rank)[::-1]))
            self.assertArrayAlmostEqual(G_copy, G.get_full_tensor())

    def test_plus(self, repeats=10):
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            H = get_random_CP_tensor(dim, np.random.randint(1, rank+1)) 
            G_plus_H = np.copy(G.get_full_tensor() + H.get_full_tensor())
            self.assertArrayAlmostEqual(G_plus_H, G.plus(H).get_full_tensor())

    def test_minus(self, repeats=10):
        for t in xrange(repeats):
            dim, rank = self.get_random_dim_and_rank()
            G = get_random_CP_tensor(dim, rank)
            H = get_random_CP_tensor(dim, np.random.randint(1, rank+1)) 
            G_minus_H = np.copy(G.get_full_tensor() + H.get_full_tensor())
            self.assertArrayAlmostEqual(G_minus_H, G.plus(H).get_full_tensor())

    def test_multiplication(self, repeats=10):
        for t in xrange(repeats):
            dim_rhs, rank_rhs = self.get_random_dim_and_rank()
            dim_result = [np.random.randint(1, 5) for i in xrange(len(dim_rhs))]
            B = get_random_CP_tensor(dim_rhs, rank_rhs)
            rank_op = np.random.randint(1, 5)
            op = CPTensorOperator([np.random.rand(dim_result[i], dim_rhs[i],
                rank_op) for i in xrange(len(dim_rhs))])
            
            # perform naive multiplication
            op_array = op.tensor.get_full_tensor()
            B_array = B.get_full_tensor()
            op_mult_B = np.zeros(dim_result)
            for i in np.ndindex(*dim_result):
                for j in np.ndindex(*dim_rhs):
                    ind_op = tuple([i[d]*dim_rhs[d]+j[d] for d in \
                        xrange(len(dim_rhs))])
                    op_mult_B[i] += op_array[ind_op] * B_array[j]
            
            self.assertArrayAlmostEqual(op_mult_B,
                    op.multiply(B).get_full_tensor())

    def runTest(self):
        self.test_get_random_CP_tensor()
        self.test_get_full_tensor()
        self.test_norm()
        self.test_normalize()
        self.test_arrange()
        self.test_plus()
        self.test_minus()
