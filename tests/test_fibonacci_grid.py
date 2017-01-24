import numpy as np
import pytest
import os

from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.model import triangulate, write_vtk
import boltons.fileutils

VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
boltons.fileutils.mkdir_p(VTK_DIR)
TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')
DECIMAL_CLOSE = 3

def test_fibonacci_grid():
    """
    Test to check whether Fibonacci_grid works
    :return:
    """

    fib_grid = FibonacciGrid()

    # Set global background grid
    radii = np.linspace(6371.0, 0.0, 5)
    resolution = np.ones_like(radii) * (6371.0 / 5)
    fib_grid.set_global_sphere(radii, resolution)
    # refinement region coarse europe
    c_min = np.radians(50)
    c_max = np.radians(120)
    l_min = np.radians(-30)
    l_max = np.radians(60)
    radii_regional = np.linspace(6371.0, 5571.0, 10)
    resolution_regional = np.ones_like(radii_regional) * (6371.0 / 20)
    fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional,
                                   angle=np.radians(-57.5), rot_y=1)

    # refinement region fine Japan
    c_min = np.radians(30)
    c_max = np.radians(75)
    l_min = np.radians(115)
    l_max = np.radians(160)
    radii_regional = np.linspace(6371.0, 5571.0, 4)
    resolution_regional = np.ones_like(radii_regional) * 200
    fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional)

    x, y, z = fib_grid.get_coordinates()
    x, y, z = fib_grid.get_coordinates(is_normalised=True)

    r, c, l = fib_grid.get_coordinates(type='spherical')
    r, c, l = fib_grid.get_coordinates(type='spherical', is_normalised=True)

    vals = np.ones_like(x)
    elements = triangulate(x, y, z)

    pts = np.array((x, y, z)).T
    np.testing.assert_allclose(np.sum(x ** 2 + y ** 2 + z ** 2), 5469.0929941834, rtol=.02, atol=0)
    write_vtk(os.path.join(VTK_DIR, 'test_fib_grid.vtk'), pts, elements, vals, 'ones')


def test_fib_grid_incorrect_type():
    fib_grid = FibonacciGrid()

    with pytest.raises(ValueError):
        fib_grid.get_coordinates(type='what?')
