from tensors.cp_tensor import CPTensor, get_random_CP_tensor
from core.compression import CompressionInfo, Compressor
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

def test(G, verbose=False, compare_parafac=False):
    print "rank of G:", G.rank
    print "G:"
    print G.get_full_tensor()
    print ""
    
    c = Compressor(accuracy=0.0001, n_iter_max=1000, 
            min_error_dec=1e-2, display_progress=verbose)
    F, info = c.compress(G)
    print "rank of F:", F.rank
    print "number of iter:", info.n_iter
    print "where the rank is increased:", info.iter_with_rank_inc
    
    print "F:"
    print F.get_full_tensor()
    print ""

    if verbose:
        print "lambdas in F:"
        print F.lambdas
        print "factors in F:" 
        for f in F.factors:
            print f
        print ""
    
    if compare_parafac:
        G_tl = tl.tensor(G.get_full_tensor())
        factors_tl = parafac(G_tl, rank=2)
        for f in factors_tl: 
            print f
        print ""
        print tl.kruskal_to_tensor(factors_tl)
        print ""

G = CPTensor([np.array([[3.0,0,4],[3,7,5]]),
    np.array([[-1,2,4],[-5,3,2],[3,4,2],[-10,-9,-8]])])

G = CPTensor([np.array([[1.0,0],[3,0]]), np.array([[5,-2],[3,-4]])])

G = CPTensor([np.array([[1.0,1],[3,-7]]), np.array([[5,-2],[3,-4]])])

G = CPTensor([np.array([[1.0]]), np.array([[1.0], [2.0]])])

G = get_random_CP_tensor([3,2,4],5)
G = G.plus(G)

test(G, verbose=True)
