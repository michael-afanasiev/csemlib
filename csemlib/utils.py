import numpy as np


def sph2cart(col, lon, rad):
    """
    Given spherical coordinates as input, returns their cartesian equivalent.
    :param col: Colatitude [radians].
    :param lon: Longitude [radians].
    :param rad: Radius.
    :return: x, y, z.
    """

    col, lon, rad = np.asarray(col), np.asarray(lon), np.asarray(rad)
    if (0 > col).any() or (col > np.math.pi).any():
        raise ValueError('Colatitude must be in range [0, pi].')

    x = rad * np.sin(col) * np.cos(lon)
    y = rad * np.sin(col) * np.sin(lon)
    z = rad * np.cos(col)

    return x, y, z


def cart2sph(x, y, z):
    """
    Given cartesian coordinates, returns their spherical equivalent.
    :param x: x.
    :param y: y.
    :param z: z.
    :return: colatitude, longitude, and radius
    """

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # Handle division by zero at the core
    with np.errstate(invalid='ignore'):
        c = np.divide(z, r)
        c = np.nan_to_num(c)

    c = np.arccos(c)
    l = np.arctan2(y, x)
    return r, c, l


def rotate(x, y, z, matrix):

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    return matrix.dot(np.array([x, y, z]))




