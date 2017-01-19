import numpy as np

from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.background.grid_data import GridData
from csemlib.utils import cart2sph

DECIMAL_CLOSE = 3


def test_griddata():
    fib_grid = FibonacciGrid()

    # Set global background grid
    radii = np.linspace(6371.0, 0.0, 5)
    resolution = np.ones_like(radii) * (6371.0 / 5)
    fib_grid.set_global_sphere(radii, resolution)
    c, l, r = cart2sph(*fib_grid.get_coordinates())

    grid_data = GridData(c, l, r, coord_system='spherical')
    grid_data.set_component('vsv', np.ones(len(r)))
    grid_data.rotate(np.radians(30), 0, 1, 0)
    np.testing.assert_almost_equal(len(r), np.sum(grid_data.get_data()), decimal=DECIMAL_CLOSE)


def test_rotation():
    """
    Test to ensure that rotation returns something meaningful
    """
    c = np.array([np.pi / 2])
    l = np.array([0.0])
    r = np.array([300.0])

    grid_data = GridData(c, l, r, coord_system='spherical')
    grid_data.rotate(np.radians(90), 0, 1, 0)
    true = np.array([np.pi, 0.0, 300.0])
    np.testing.assert_almost_equal(true, grid_data.get_coordinates(coordinate_type='spherical')[0], decimal=DECIMAL_CLOSE)

    c = np.array([np.pi / 2])
    l = np.array([0.0])
    r = np.array([300.0])

    grid_data = GridData(c, l, r, coord_system='spherical')
    grid_data.rotate(np.radians(180), 0, 0, 1)
    true = np.array([np.pi / 2, np.pi, 300.0])
    np.testing.assert_almost_equal(true, grid_data.get_coordinates(coordinate_type='spherical')[0], decimal=DECIMAL_CLOSE)


def test_add_and_del_components():
    """
    Test whether add and delete components work as they should

    """
    a = np.linspace(0, 3, 4)
    x, y, z = np.meshgrid(a, a, a)

    grid_data = GridData(x.flatten(), y.flatten(), z.flatten(), coord_system='cartesian')
    grid_data.add_components(['vsv', 'vsh', 'rho'])
    grid_data.del_components(['vsv', 'rho'])

    if grid_data.components != ['vsh']:
        raise AssertionError()
