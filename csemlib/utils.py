import numpy as np


def sph2cart(col, lon, rad=1):
    """
    Given spherical coordinates as input, returns their cartesian equivalent.
    :param col: Colatitude [radians].
    :param lon: Longitude [radians].
    :param rad: Radius.
    :return: x, y, z.
    """

    if 0 > col or col > np.math.pi:
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

    r = np.math.sqrt(x ** 2 + y ** 2 + z ** 2)
    # Break obviously if radius is zero. In that case we're just
    # at the center of the Earth.
    try:
        c = np.math.acos(z / r)
    except ZeroDivisionError:
        return 0, 0, 0
    l = np.math.atan2(y, x)

    return r, c, l
