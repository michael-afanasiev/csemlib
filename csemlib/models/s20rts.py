import numpy as np
from scipy.special import sph_harm

def eval_sph_harm():
    coeffs = np.array([[0.1543e-1],
                       [0.1590E-01, -0.1336E-01, 0.3469E-02]])

    val = sph_harm(1, 1, 0, 0)
    print(val)
