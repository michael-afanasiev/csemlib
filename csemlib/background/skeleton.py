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


def multiple_fibonacci_spheres(radii, n_samples):
    """
    This function returns the fibonacci sphere coordinates for multiple layers
    :param radii: array describing this distances from the core in km, ordered from shallow to deep?
    :return layer index:
    """
    num_layers = len(radii)
    pts = np.array(fibonacci_sphere(n_samples))
    all_layers = pts

    if num_layers > 1:
        for rad in radii[1:]:
            all_layers = np.append(all_layers, pts * rad/radii[0], axis=1)

    return all_layers
