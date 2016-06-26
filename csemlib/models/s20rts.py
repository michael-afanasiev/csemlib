import io
import os

import numpy as np
from scipy.special import sph_harm

from csemlib.models.model import Model


class S20rts(Model):
    """
    Class handling S20rts evaluations.
    """

    def data(self):
        pass

    def __init__(self):
        super(S20rts, self).__init__()
        directory, _ = os.path.split(os.path.split(__file__)[0])
        self.directory = os.path.join(directory, 'data', 's20rts')

    def read(self):
        coeff_file = os.path.join(self.directory, 'S20RTS.dat')
        with io.open(coeff_file, 'rt') as fh:
            coeffs = np.asarray(fh.read().split(), dtype=float)

        tot = 0
        for s in range(21):
            setattr(self, 'l%d' % s, [])
            for i in range(21):
                c = []
                for j in range(2 * i + 1):
                    c.append(coeffs[tot])
                    tot += 1
                getattr(self, 'l%d' % s).append(np.array(c))

    def write(self):
        pass

    def eval(self, x, y, z, param):

        val = np.zeros_like(x)
        for n in range(20):
            imag, real = 1, 0
            for m in range(2 * n + 1):
                if m == 0 or (m % 2):
                    val += self.l0[n][m] * np.real(sph_harm(real, n, y, x))
                    real += 1
                else:
                    val += self.l0[n][m] * np.imag(sph_harm(imag, n, y, x))
                    imag += 1

        return val
