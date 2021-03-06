import tensorly as tl
import numpy as np
from copy import deepcopy

class CPTensor:
    """ A light wrapper around list of NDArrays to represent CP tensors.
        If G is a rank R tensor of dim I1 x I2 x ... x In,
        the list of NDArrays in factors are of shape
        I1 x R, I2 x R, ... , In x R. 
    """
    def __init__(self, factors, lambdas=None):
        # representation of a zero tensor
        if factors == []:
            self.rank = 0
            self.lambdas = []
            self.dim = []
            return

        # use copy to make sure that a new list instance is created
        self.factors = deepcopy(factors)
        try:
            self.rank = factors[0].shape[1]
        except IndexError:
            print "factors must be a list of 2D arrays"
            raise IndexError

        if lambdas is None:
            self.lambdas = np.ones(self.rank)
        else:
            assert type(lambdas) is np.ndarray
            assert lambdas.shape == (self.rank, )
            # use copy to make sure that a new instance is created
            self.lambdas = np.copy(lambdas)
            # flip signs to make sure lambdas are all positive
            for i in xrange(self.rank):
                if self.lambdas[i] < 0:
                    self.factors[0][:, i] *= -1.0
                    self.lambdas[i] = -self.lambdas[i]

        self.dim = []
        for f in factors:
            assert f.shape[1] == self.rank, "wrong array dimensions"
            self.dim.append(f.shape[0])

    def get_full_tensor(self):
        return tl.kruskal_to_tensor(
                self.factors[:-1] + \
                [self.factors[-1] * self.lambdas.reshape(1,-1)])

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
            self.lambdas = self.lambdas * scale
            # avoid division by 0 (those lambdas = 0 by the line above)
            scale[scale == 0] = 1.0
            self.factors[i] = self.factors[i] / scale.reshape(1,-1)

    def distribute_lambda(self):
        """ Distributes the lambdas evenly across all factors.
        """
        assert np.all(self.lambdas >= 0)
        scaling = np.diag(np.power(self.lambdas, (1.0/len(self.dim))))
        for i in xrange(len(self.factors)):
            self.factors[i] = np.dot(self.factors[i], scaling)
        self.lambdas = np.ones(self.rank)

    def arrange(self):
        """ Normalizes the columns of the factor matrices and then sorts the
        tensor components by magnitude, greatest to least.
        """
        self.normalize()
        perm = np.argsort(self.lambdas)[::-1]
        self.lambdas = self.lambdas[perm]
        for i in xrange(len(self.factors)):
            self.factors[i] = self.factors[i][:,perm]

    def plus(self, B):
        """ Computes self + B, where B is a CPTensor of the same dim """
        assert isinstance(B, CPTensor), "wrong type, not a CPTensor"
        if B.rank == 0:
            return self
        if self.rank == 0:
            return B
        assert np.array_equal(self.dim, B.dim), "tensor dimension mismatch"
        concat_factors = [np.concatenate(
            (self.factors[i], B.factors[i]), axis=1) \
                    for i in xrange(len(self.factors))]
        return CPTensor(concat_factors,
                np.concatenate((self.lambdas, B.lambdas)))

    def minus(self, B):
        """ Computes self - B, where B is a CPTensor of the same dim """
        return self.plus(CPTensor(B.factors, -B.lambdas))
    
    def display(self):
        print "CPTensor with the following factors:"
        for f in self.factors:
            print f
        print "and lambdas:"
        print self.lambdas
        print ""

    def getNormalizationRatio(self):
        """ Returns the ratio of the smallest normalization constant divided by
        the largest normalization constant
        """
        self.arrange()
        return self.lambdas[-1]/self.lambdas[0]

class CPTensorOperator:
    """ A wrapper around CPTensor to represent operators.
        If A is a rank R tensor operator mapping tensors of dim I1 x ... x In to
        tensors of dim I1' x ... x In',
        the corresponding CPTensor object has dim (I1' x I1) x ... x (In' x In).
    """
    def __init__(self, factors, lambdas=None):
        try:
            self.rank = factors[0].shape[2]
        except IndexError:
            print "factors must be a list of 3D arrays"
            raise IndexError

        factors_flat = []
        self.dim = []
        for f in factors:
            assert len(f.shape) == 3 and f.shape[2] == self.rank, \
                "factors must be a list of 3D arrays of dimension I_d' x I_d x R"
            self.dim.append((f.shape[0], f.shape[1]))
            factors_flat.append(f.reshape(f.shape[0] * f.shape[1], self.rank))

        self.tensor = CPTensor(factors_flat, lambdas)

    def multiply(self, B):
        """ B must be a tensor of dim I1 x ... x In
            The multiplication results in a tensor of dim I1' x ... x In'
        """
        assert isinstance(B, CPTensor), "wrong type, not a CPTensor"
        assert np.array_equal([t[1] for t in self.dim], B.dim), \
            "dimensions do not match"
        factors = []
        for d in xrange(len(self.dim)):
            f = np.zeros((self.dim[d][0], self.rank * B.rank))
            for i in range(self.rank):
                operator = self.tensor.factors[d][:,i].reshape(self.dim[d]),
                f[:, i*B.rank:(i+1)*B.rank] = np.dot(operator, B.factors[d])
            factors.append(f)
        lambdas = np.outer(self.tensor.lambdas, B.lambdas).flatten()
        return CPTensor(factors, lambdas)

    def display(self):
        print "CPTensorOperator with the following factors:"
        for i, f in enumerate(self.tensor.factors):
            print f.reshape(self.dim[i][0], self.dim[i][1])
        print "and lambdas:"
        print self.tensor.lambdas
        print ""

def get_random_CP_tensor(dim, rank):
    """ Generates a random CP tensor of the specified rank and dimensions.
    Each generated factor matrix is unit-norm.
    """
    factors = [np.random.rand(d, rank) for d in dim]
    # TODO: is this normalization most suited?
    factors = [f / np.linalg.norm(f) for f in factors]
    return CPTensor(factors)

def get_random_CP_tensor_operator(dim, rank):
    """ Generates a random CP tensor of the specified rank and dimensions.
    Each generated factor matrix is unit-norm.
    """
    factors = [np.random.rand(d[0], d[1], rank) for d in dim]
    return CPTensorOperator(factors)

def get_identity_tensor_operator(dim):
    factors = [np.eye(d).reshape(d,d,1) for d in dim]
    return CPTensorOperator(factors)
