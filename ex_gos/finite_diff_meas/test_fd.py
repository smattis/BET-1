import numpy as np
from average_qoi import *

A = lb_model5(np.array([[1.2, 1.3]]))

B = lb_model_exact(np.array([[1.2, 1.3]]))

print A[1]/(B[0]-A[0])

import pdb
pdb.set_trace()
