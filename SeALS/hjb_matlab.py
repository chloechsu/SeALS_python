from core.info import ALSOptions
from core.least_squares import Solver
from tensors.matlab_interface import load_cp_tensor, load_cp_tensor_operator, \
        save_cp_tensor
import sys
import numpy as np

""" 
This script takes the tensor operator and boundary conditions from MATLAB, and
solves for op*F = bc. The result is saved in a new .mat file.

To run ALS: 
./hjb_script_with_matlab filename_op filename_bc output_filename use_SeALS

If use_SeALS is true, use the sequential variant. Otherwise, use vanilla ALS.
"""

if len(sys.argv) < 5:
    print "To run ALS:"
    print "./hjb_script_with_matlab" + \
            "filename_op filename_bc output_filename use_SeALS"
    sys.exit()

op = load_cp_tensor_operator(sys.argv[1])
bc = load_cp_tensor(sys.argv[2])
als_options = ALSOptions(accuracy=1e-9, max_rank=30, verbose=True)
assert sys.argv[4].lower() == 'true' or sys.argv[4].lower() == 'false'
use_seals = (sys.argv[4].lower() == 'true')
solver = Solver(als_options, use_SeALS=use_seals)
F, info = solver.solve(op, bc)
save_cp_tensor(F, sys.argv[3])
