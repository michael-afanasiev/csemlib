import math

import numpy as np
from csemlib.models.model import triangulate, write_vtk
from csemlib.utils import cart2sph


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
    :return layer index:
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
    :return layer index:
    """
    r_earth = 6371.0
    num_layers = len(radii)
    n_samples = int(((4 * radii[0] ** 2) / (resolution ** 2))) + 1
    if n_samples < min_samples:
        n_samples = min_samples
    pts = np.array(fibonacci_sphere(n_samples))
    all_layers = pts * radii[0]
    print(np.shape(all_layers))

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
        print(np.shape(all_layers))





    return all_layers



def fibonacci_sphere_14_vec(n_samples):


    # Define Refinement region
    c_max = np.radians(40)
    c_min = np.radians(70)
    z_start = math.cos(c_max)
    z_end = math.cos(c_min)
    z_len = z_end - z_start

    l_min = np.radians(125)
    l_max = np.radians(155)
    part_of_circle = (l_max - l_min) / (2 * math.pi)

    golden_angle = math.pi * (3.0 - math.sqrt(5.0)) * part_of_circle


    def populate(idx):
        offset = z_len / n_samples
        z = ((idx * offset) + z_start) + (offset / 2.0)
        r = math.sqrt(1 - z ** 2)
        phi = (idx * golden_angle) % (part_of_circle * 2 * math.pi) + l_min
        x = math.cos(phi) * r
        y = math.sin(phi) * r

        return x, y, z

    g = np.vectorize(populate)
    return np.fromfunction(g, (n_samples,))


def get_multiple_layers(x, y, z, radii):
    r_earth = 6371.0
    all_x = np.zeros(0)
    all_y = np.zeros(0)
    all_z = np.zeros(0)

    for rad in radii:
        if rad == 0:
            all_x = np.append(all_x, np.zeros(1))
            all_y = np.append(all_y, np.zeros(1))
            all_z = np.append(all_z, np.zeros(1))
            continue

        rel_rad = rad / r_earth
        new_x = x * rel_rad
        new_y = y * rel_rad
        new_z = z * rel_rad

        all_x = np.append(all_x, new_x)
        all_y = np.append(all_y, new_y)
        all_z = np.append(all_z, new_z)
    return all_x, all_y, all_z

def s20rts_vtk_single_sphere():
    """
    Test to ensure that a vtk of s20rts is written succesfully.
    :return:


    """

    x_g, y_g, z_g = fibonacci_sphere(100)
    global_radii = np.linspace(0, 6371.0, 5)
    x_g, y_g, z_g = get_multiple_layers(x_g, y_g, z_g, global_radii)
    x, y, z = fibonacci_sphere_14_vec(100)
    radii = np.linspace(5500.0, 6371.0, 10)
    x, y, z = get_multiple_layers(x, y, z, radii)
    x = np.append(x, x_g)
    y = np.append(y, y_g)
    z = np.append(z, z_g)


    # x = x_g
    # y = y_g
    # z = z_g


    vals = np.ones_like(x)

    elements = triangulate(x,y,z)

    pts = np.array((x, y, z)).T

    write_vtk("test_vec.vtk", pts, elements, vals, 'vs')

s20rts_vtk_single_sphere()