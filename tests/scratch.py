import os
import numpy as np

from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.background.grid_data import GridData
from csemlib.models.crust import Crust
from csemlib.models.model import triangulate, write_vtk
from csemlib.models.one_dimensional import prem_eval_point_cloud
from csemlib.models.s20rts import S20rts
from csemlib.models.ses3d import Ses3d
from csemlib.models.ses3d_rbf import Ses3d_rbf
from csemlib.utils import cart2sph

##########################################################################
#  Generate visualisation grid
fib_grid = FibonacciGrid()
# Set global background grid
radii = np.linspace(6271.0, 6271.0, 1)
resolution = np.ones_like(radii) * (6371.0 - 5771.0) / 2
fib_grid.set_global_sphere(radii, resolution)

# Make regional refinement
c_min = np.radians(50)
c_max = np.radians(120)
l_min = np.radians(-30)
l_max = np.radians(60)
radii_regional = np.linspace(6271.0, 6271.0, 1)
resolution_regional = np.ones_like(radii_regional) * 80
fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional)


# Setup grid_data
grid_data = GridData(*fib_grid.get_coordinates())
grid_data.rotate(np.radians(-57.5), 0, 1, 0)

# # Evaluate Prem
# rho, vpv, vsv, vsh = prem_eval_point_cloud(grid_data.df['r'])
grid_data.set_component('vsv', np.ones(len(grid_data)))
#
# # Add S20rts
# s20 = S20rts()
# s20.eval_point_cloud_griddata(grid_data)
#
# # Add crust and topography
# cst = Crust()
# cst.eval_point_cloud_grid_data(grid_data)
#
# # Add ses3d model
# mod = Ses3d_rbf('japan', os.path.join('/home/sed/CSEM/csemlib/tests/test_data', 'japan_new_format'), components=grid_data.components)
# mod.eval_point_cloud_griddata(grid_data)


# # Add ses3d model
# mod = Ses3d_rbf('europe', os.path.join('/home/sed/CSEM/csemlib/ses3d_models', 'europe_1s'),
#             components=grid_data.components)
# #mod.eval_point_cloud_griddata(grid_data)
#
#
# x, y, z = mod.grid_data_ses3d.get_coordinates(coordinate_type='cartesian').T
x, y, z = grid_data.get_coordinates(coordinate_type='cartesian').T
dat = grid_data.get_component('vsv')
# dat = mod.grid_data_ses3d.df['vsv']
elements = triangulate(x, y, z)
#
pts = np.array((x, y, z)).T
write_vtk("ses3d_griddata_all_regional.vtk", pts, elements, dat, 'ses3dvsv')