import math

import numpy as np


def fibonacci_sphere(n_samples):
    def populate(idx):
        offset = 2.0 / n_samples
        increment = math.pi * (3.0 - math.sqrt(5.0))
        y = ((idx * offset) - 1) + (offset / 2.0)
        r = math.sqrt(1 - y ** 2)
        phi = (idx % n_samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        return x, y, z

    g = np.vectorize(populate)
    return np.fromfunction(g, (n_samples,))
