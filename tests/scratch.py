
##########################################################################
#  Generate visualisation grid
import os

import numpy as np

from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.background.grid_data import GridData
from csemlib.models.crust import Crust
from csemlib.models.model import triangulate, write_vtk
from csemlib.models.one_dimensional import prem_eval_point_cloud
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d import Ses3d
from csemlib.utils import cart2sph



fib_grid = FibonacciGrid()
# Set global background grid
radii = np.linspace(6250.0, 0.0, 30)
resolution = np.ones_like(radii) * (6371.0 / 30)

fib_grid.set_global_sphere(radii, resolution)

c_min = np.radians(25)
c_max = np.radians(75)
l_min = np.radians(110)
l_max = np.radians(175)
radii_regional = np.linspace(6250.0, 6250.0, 1)
resolution_regional = np.ones_like(radii_regional) * 50
fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional)

x, y, z = fib_grid.get_coordinates()
pts_new = np.array((x, y, z)).T

c, l, r = cart2sph(x, y, z)

# Evaluate Prem
rho, vpv, vsv, vsh = prem_eval_point_cloud(r)

# Setup grid_data
grid_data = GridData(x, y, z)
grid_data.add_col_lon_rad()
grid_data.set_component('vsv', np.ones(len(vsv)))


# # Add S20rts
# s20 = S20rts()
# s20.eval_point_cloud_griddata(grid_data)
#
# # Add crust and topography
# cst = Crust()
# cst.eval_point_cloud_grid_data(grid_data)

# Add regional model
mod = Ses3d('japan', os.path.join('/home/sed/CSEM/csemlib/tests/test_data', 'japan_new_format'))
mod.eval_point_cloud_griddata(grid_data)

x, y, z = grid_data.get_coordinates(coordinate_type='cartesian').T
dat = grid_data.df['vsv'].values
elements = triangulate(x, y, z)

pts = np.array((x, y, z)).T
write_vtk("ses3d_griddata_all_regional.vtk", pts, elements, dat, 'ses3dvsv')