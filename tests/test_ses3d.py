import os

import numpy as np
import csemlib.background.skeleton as skl

import csemlib.models.ses3d as s3d
from csemlib.background.grid_data import GridData
from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.crust import Crust
from csemlib.models.model import triangulate, write_vtk
from csemlib.models.one_dimensional import prem_eval_point_cloud
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d_rbf import Ses3d_rbf
from csemlib.utils import cart2sph

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')
VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
DECIMAL_CLOSE = 3


def test_ses3d():

    #  Generate visualisation grid
    fib_grid = FibonacciGrid()
    # Set global background grid
    radii = np.linspace(6271.0, 6271.0, 1)
    resolution = np.ones_like(radii) * (6371.0 / 15)
    fib_grid.set_global_sphere(radii, resolution)
    # refinement region coarse
    c_min = np.radians(35)
    c_max = np.radians(65)
    l_min = np.radians(125)
    l_max = np.radians(155)
    radii_regional = np.linspace(6271.0, 6271.0, 1)
    resolution_regional = np.ones_like(radii_regional) * 50
    fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional)


    # Setup GridData
    grid_data = GridData(*fib_grid.get_coordinates())

    # Evaluate Prem
    rho, vpv, vsv, vsh = prem_eval_point_cloud(grid_data.df['r'])
    grid_data.set_component('vsv', np.ones(len(grid_data)))

    mod = Ses3d_rbf('japan', os.path.join('/home/sed/CSEM/csemlib/tests/test_data', 'japan'),
                    components=grid_data.components, interp_method='nearest_neighbour')
    mod.eval_point_cloud_griddata(grid_data)

    # Write vtk
    x, y, z = grid_data.get_coordinates(coordinate_type='cartesian').T
    elements = triangulate(x, y, z)
    pts = np.array((x, y, z)).T
    write_vtk("ses3d_nearest_neighbour.vtk", pts, elements, grid_data.get_component('vsv'), 'ses3dvsv')
