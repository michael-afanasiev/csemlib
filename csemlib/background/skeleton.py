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


def multiple_fibonacci_spheres(radii, n_samples, normalized_radius=True):
    """
    This function returns the fibonacci sphere coordinates for multiple layers
    :param radii: array describing distances from the core in km, ordered from shallow to deep
    :param n_samples: amount of samples on the fibonacci sphere
    :return all_layers: These are the x, y, z coordinates of all layers
    """
    r_earth = 6371.0
    num_layers = len(radii)
    pts = np.array(fibonacci_sphere(n_samples))
    if normalized_radius:
        all_layers = pts * radii[0] / r_earth
    else:
        all_layers = pts * radii[0]

    if num_layers > 1:
        for rad in radii[1:]:
            # if requested radius is equal to zero, add one point in the center
            if rad == 0.0:
                all_layers = np.append(all_layers, np.zeros((3,1)), axis=1)
                continue

            if normalized_radius:
                all_layers = np.append(all_layers, pts * rad/r_earth, axis=1)
            else:
                all_layers = np.append(all_layers, pts * rad, axis=1)

    return all_layers

def multiple_fibonacci_resolution(radii, resolution=200.0, min_samples=10):
    """
    This function returns the fibonacci sphere coordinates for multiple layers
    :param radii: array describing distances from the core in km, ordered from shallow to deep
    :param resolution: average distance to the next sample in km
    :param min_samples: minimum amount of samples on the fibonacci sphere
    :return all_layers: These are the x, y, z coordinates of all layers
    """
    n_samples = int(((4 * radii[0] ** 2) / (resolution ** 2))) + 1
    if n_samples < min_samples:
        n_samples = min_samples
    pts = np.array(fibonacci_sphere(n_samples))
    all_layers = pts * radii[0]

    for rad in radii[1:-1]:
        if rad == 0.0:
            all_layers = np.append(all_layers, np.zeros((3,1)), axis=1)
            continue
        n_samples = int((4 * rad ** 2) / (resolution ** 2)) + 1
        if n_samples < min_samples:
            n_samples = min_samples

        pts = np.array(fibonacci_sphere(n_samples))
        layer = pts * rad
        all_layers = np.append(all_layers, layer, axis=1)

    return all_layers
