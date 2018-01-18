import tensorly as tl
import numpy as np

class CPTensor:
    """ A light wrapper around list of NDArrays to represent CP tensors.
        If G is a rank R tensor of dim I1 x I2 x ... x In,
        the list of NDArrays in factors are of shape
        I1 x R, I2 x R, ... , In x R. 
    """
    def __init__(self, factors, lambdas=None):
        self.factors = factors
        self.rank = factors[0].shape[1]
        if lambdas is None:
            self.lambdas = np.ones(self.rank)
        else:
            assert type(lambdas) is np.ndarray
            assert lambdas.shape == (self.rank, )
            self.lambdas = lambdas
        self.dim = []
        for f in factors:
            assert f.shape[1] == self.rank, "wrong array dimensions"
            self.dim.append(f.shape[0])

    def norm(self):
        """ Returns the Frobenius norm of a CP-tensor """
        # Compute the matrix of correlation coefficients
        coef_matrix = np.outer(self.lambdas, self.lambdas)
        for f in self.factors:
            coef_matrix *= np.dot(f.transpose(), f)
        return np.sqrt(np.abs(coef_matrix.sum()))

    def normalize(self):
        """ Normalizes the columns of each factor matrix using the vector
        2-norm, absorbing the excess weight into lambda. Also ensures that
        lambda is positive.
        """
        for i in xrange(len(self.factors)):
            scale = np.linalg.norm(self.factors[i], axis=0)
            self.lambdas *= scale
            # TODO: handle zeros in scale more carefully
            self.factors[i] /= scale

    def arrange(self):
        """ Normalizes the columns of the factor matrices and then sorts the
        tensor components by magnitude, greatest to least.
        """
        self.normalize()
        perm = np.argsort(self.lambdas)[::-1]
        self.lambdas = self.lambdas[perm]
        for i in xrange(len(self.factors)):
            self.factors[i] = self.factors[i][:,perm]

    def add(self, B):
        """ Computes self + B, where B is a CPTensor of the same dim """
        assert type(B) is CPTensor, "wrong type, not a CPTensor"
        assert np.array_equal(self.dim, B.dim), "tensor dimension mismatch"
        concat_factors = [np.concatenate(
            self.factors[i], B.factors[i], axis=1) \
                    for i in xrange(len(self.factors))]
        return CPTensor(concat_factors,
                np.concatenate(self.lambdas, B.lambdas))

    def minus(self, B):
        """ Computes self - B, where B is a CPTensor of the same dim """
        B.lambdas = -B.lambdas
        return self.add(B)

def get_random_CP_tensor(dim, rank):
    """ Generates a random CP tensor of the specified rank and dimensions.
    Each generated factor matrix is unit-norm.
    """
    factors = [np.random.rand(d, rank) for d in dim]
    # TODO: is this normalization most suited?
    factors = [f / np.linalg.norm(f) for f in factors]
    return CPTensor(factors)
