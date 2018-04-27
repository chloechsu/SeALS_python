import numpy as np
from scipy.io import loadmat, savemat
import sys
from tensors.cp_tensor import CPTensor, CPTensorOperator

def load_cp_tensor(filename):
    """ Load a tensor from .mat file.
        The .mat file represents a MATLAB struct, with lambda and factors as
        fields in the struct.   
    """
    try:
        mat = loadmat(filename)
    except:
        print("cannot parse input file, error:", sys.exc_info()[0])
    # MATLAB stores the factors as a ndim x 1 cell array for ktensor
    factors = list([f.astype('double') for f in mat['factors'][:,0]])
    return CPTensor(factors, lambdas=mat['lambdas'][:,0].astype('double'))

def load_cp_tensor_operator(filename):
    """ Load a tensor operator from .mat file.
        The .mat file represents a MATLAB struct, with lambda and factors as
        fields in the struct.   
        Each factor is a 2D array with shape (n^2) * r.
    """
    try:
        mat = loadmat(filename)
    except:
        print("cannot parse input file, error:", sys.exc_info()[0])

    # MATLAB stores the factors as a 1 x ndim cell array for ktensor operator
    # (note this is different from ktensor)
    factors = list([f.astype('double') for f in mat['factors'][0,:]])
    d = int(np.sqrt(factors[0].shape[0])) 
    # Check the first dimension for each factor are perfect squares
    for x in factors:
        assert d ** 2 == x.shape[0], \
                "each factor must have shape (d^2) x r"
    # Note that in matlab the index for tensor operator is col first, so we need
    # to swap axes
    factors = [np.swapaxes(x.reshape(d,d,-1),0,1) for x in factors]
    return CPTensorOperator(factors,
            lambdas=mat['lambdas'][:,0].astype('double'))

def save_cp_tensor(tensor, filename):
    """ Save a tensor to .mat file.
        The .mat file represents a MATLAB struct, with lambda and factors as
        fields in the struct.   
    """
    # To get the correct matlab cell array format, need array of array
    cell_array = np.empty(len(tensor.factors), dtype='object')
    for i in xrange(len(tensor.factors)):
        cell_array[i] = tensor.factors[i]
    dct = {
            'factors': cell_array, 
            'lambdas': np.array(tensor.lambdas)
          }
    savemat(filename, dct, oned_as='column')
