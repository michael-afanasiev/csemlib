import os

import numpy as np
import csemlib.background.skeleton as skl

import csemlib.models.ses3d as s3d
from csemlib.background.grid_data import GridData
from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.model import triangulate, write_vtk
from csemlib.models.one_dimensional import prem_eval_point_cloud
from csemlib.models.ses3d_rbf import Ses3d_rbf
from csemlib.utils import cart2sph

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')
VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
DECIMAL_CLOSE = 3



#
# def fail_ses3d():
#     mod = s3d.Ses3d('japan', os.path.join(TEST_DATA_DIR, 'japan'),
#                     components=['drho', 'dvsv', 'dvsh', 'dvp'])
#     mod.read()
#
#     # Generate Fibonacci sphere at 20 km depth
#     x, y, z = skl.fibonacci_sphere(100000)
#     x *= 6250.0
#     y *= 6250.0
#     z *= 6250.0
#
#     # Eval ses3d
#     interp = mod.eval(x, y, z, param=['dvsv', 'drho', 'dvsh', 'dvp'])
#
#     # Write to vtk
#     elements = triangulate(x, y, z)
#     pts = np.array((x, y, z)).T
#     write_vtk(os.path.join(VTK_DIR, 'ses3d_fail.vtk'), pts, elements, interp[:,0], 'ses3d')
#
# fail_ses3d()
#



#  Generate visualisation grid
fib_grid = FibonacciGrid()
# Set global background grid
radii = np.linspace(6371.0, 0, 15)
resolution = np.ones_like(radii) * (6371.0 / 15)
fib_grid.set_global_sphere(radii, resolution)
# refinement region coarse
c_min = np.radians(35)
c_max = np.radians(65)
l_min = np.radians(125)
l_max = np.radians(155)
radii_regional = np.linspace(6371.0, 5771.0, 14)
resolution_regional = np.ones_like(radii_regional) * 50
fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional)
x, y, z = fib_grid.get_coordinates()
pts_new = np.array((x, y, z)).T

c, l, r = cart2sph(x, y, z)

print(len(r))
# Evaluate Prem
rho, vpv, vsv, vsh = prem_eval_point_cloud(r)

components = ['rho', 'vp', 'vsv', 'vsh']
grid_data = GridData(x, y, z, components=components)

#grid_data.set_component('vsv', np.ones(len(grid_data)))
grid_data.set_component('vsv', vsv)

mod = Ses3d_rbf('japan', os.path.join('/home/sed/CSEM/csemlib/tests/test_data', 'japan'),
                components=['dvsv'])

grid_data = mod.eval_point_cloud(grid_data)


x, y, z = grid_data.get_coordinates(coordinate_type='cartesian').T
dat = grid_data.df['vsv'].values
elements = triangulate(x, y, z)
# #print(np.shape(pts_hull))

pts = np.array((x, y, z)).T
write_vtk("ses3d_griddata_rbf.vtk", pts, elements, dat, 'ses3dvsv')