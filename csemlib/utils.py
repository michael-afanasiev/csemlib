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
    return c, l, r


def get_rot_matrix(angle, x, y, z):
    """
    :param angle: Rotation angle in radians (Right-Hand rule, counterclockwise positive)
    :param x: x-component of rotational vector
    :param y: y-component of rotational vector
    :param z: z-component of rotational vector
    :return: Rotational Matrix
    """
    # Normalize vector.
    norm = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    x /= norm
    y /= norm
    z /= norm

    # Setup matrix components.
    matrix = np.empty((3, 3))
    matrix[0, 0] = np.cos(angle) + (x ** 2) * (1 - np.cos(angle))
    matrix[1, 0] = z * np.sin(angle) + x * y * (1 - np.cos(angle))
    matrix[2, 0] = (-1) * y * np.sin(angle) + x * z * (1 - np.cos(angle))
    matrix[0, 1] = x * y * (1 - np.cos(angle)) - z * np.sin(angle)
    matrix[1, 1] = np.cos(angle) + (y ** 2) * (1 - np.cos(angle))
    matrix[2, 1] = x * np.sin(angle) + y * z * (1 - np.cos(angle))
    matrix[0, 2] = y * np.sin(angle) + x * z * (1 - np.cos(angle))
    matrix[1, 2] = (-1) * x * np.sin(angle) + y * z * (1 - np.cos(angle))
    matrix[2, 2] = np.cos(angle) + (z * z) * (1 - np.cos(angle))

    return matrix

def rotate(x, y, z, matrix):
    """

    :param x: x-coordinates to be rotated
    :param y: y-coordinates to be rotated
    :param z: z-coordinates to be rotated
    :param matrix: Rotational matrix obtained from get_rot_matrix
    :return: Rotated x,y,z coordinates
    """

    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)
    return matrix.dot(np.array([x, y, z]))




