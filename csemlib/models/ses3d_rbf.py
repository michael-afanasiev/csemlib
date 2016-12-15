import datetime
import io
import os

import numpy as np
import xarray


from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.model import Model, shade, triangulate, interpolate, write_vtk
from csemlib.utils import sph2cart, rotate, cart2sph
from csemlib.models.ses3d import Ses3d


import numpy as np
import scipy.spatial as spatial
from scipy.interpolate import Rbf
from scipy.spatial.qhull import ConvexHull
from scipy.interpolate import griddata

# from .model import Model, shade, triangulate, interpolate
# from ..utils import sph2cart, rotate



class Ses3d_rbf(Ses3d):
    """
    Class handling file-IO for a model in SES3D format.
    """

    def __init__(self, name, directory, components=[],
                 rotation_vector=None, rotation_angle=None, doi=None):
        super(Ses3d_rbf, self).__init__(name, directory, components,
                 rotation_vector, rotation_angle, doi)
        self.read()


    def split_domain(self, pts_original, pts_new):
        hull = ConvexHull(pts_original)
        pts_hull = []
        for point in np.unique(hull.simplices.flatten()):
            pts_hull.append(pts_original[point])
        pts_hull = np.array(pts_hull)

        # Generate convex hull of original points and only continue with points that fall inside
        hull = spatial.Delaunay(pts_hull)

        # Split points into points that fall inside and outside of convex hull
        in_or_out = hull.find_simplex(pts_new)>=0
        indices_in = np.where(in_or_out == True)
        indices_out = np.where(in_or_out == False)
        pts_other = pts_new[indices_out]
        pts_new = pts_new[indices_in]

        return pts_new, pts_other


    def get_data_pts_model(self):
        model_coords = np.zeroes(len(self.data['x'].values.ravel()), 3)
        model_coords[:,0] = self.data['x'].values.ravel()
        model_coords[:,1] = self.data['y'].values.ravel()
        model_coords[:,2] = self.data['z'].values.ravel()

        model_data = np.zeroes(len(self.data['x'].values.ravel()), len(self.components))

        index = 0
        for component in self.components:
            model_data[:, index] = self.data[component].values.ravel
            index += 1

        return model_coords, model_data

    def eval_point_cloud(self,c, l, r, rho, vpv, vsv, vsh):
        model_coords, model_data = self.get_data_pts_model()
        dat_old = np.array(rho, vpv, vsv, vsh)
        x, y, z = sph2cart(c, l, r)
        pts_new = np.array((x, y, z)).T

        pts_original = model_coords
        data = model_data

        pts_new, pts_other = self.split_domain(pts_original, pts_new)

        # Generate KDTrees
        pnt_tree_orig = spatial.cKDTree(pts_original)

        # Method based on nearest points:
        _, pairs = pnt_tree_orig.query(pts_new, k=6)

        i = 0
        dat_new = np.zeros(np.shape(pts_new)[0])
        for idx in pairs:
            x_c_orig, y_c_orig, z_c_orig = pts_original[idx].T
            dat_orig = data[idx]

            rbfi = Rbf(x_c_orig, y_c_orig, z_c_orig, dat_orig)
            x_c_new, y_c_new, z_c_new = pts_new[i]
            dat_new[i] = rbfi(x_c_new, y_c_new, z_c_new)

            i += 1
            if i % 200 == 0:
                print(i)

        pts_all = np.append(pts_new, pts_other, axis=0)
        dat_all = np.append(dat_new, np.zeros_like(pts_other[:,0]))

        return pts_all, dat_all



def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull
    #return hull.find_simplex(p)>=0





# Get original data (this seems to work fine)
mod = Ses3d_rbf('japan', os.path.join('/home/sed/CSEM/csemlib/tests/test_data', 'japan'),
                components=['drho', 'dvsv', 'dvsh', 'dvp'])
# mod.read()

d = mod.data['dvsv'].values.ravel()
x = mod.data['x'].values.ravel()
y = mod.data['y'].values.ravel()
z = mod.data['z'].values.ravel()

r, _, l = cart2sph(x, y, z)
pts_original = np.array((x, y, z)).T
data = np.array(d)
#data = np.degrees(l)


#  Generate visualisation grid
fib_grid = FibonacciGrid()
# Set global background grid
radii = np.linspace(6250.0, 0.0, 15)
resolution = np.ones_like(radii) * (6350.0 / 15)
fib_grid.set_global_sphere(radii, resolution)
# refinement region coarse
c_min = np.radians(30)
c_max = np.radians(70)
l_min = np.radians(120)
l_max = np.radians(160)
radii_regional = np.linspace(6250.0, 6150.0, 4)
resolution_regional = np.ones_like(radii_regional) * 50
fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional)
x, y, z = fib_grid.get_coordinates()
pts_new = np.array((x, y, z)).T



hull = ConvexHull(pts_original)
pts_hull = []
for point in np.unique(hull.simplices.flatten()):
    pts_hull.append(pts_original[point])
pts_hull = np.array(pts_hull)

# Generate convex hull of original points and only continue with points that fall inside
hull = spatial.Delaunay(pts_hull)

# Split points into points that fall inside and outside of convex hull
in_or_out = hull.find_simplex(pts_new)>=0
indices_in = np.where(in_or_out == True)
indices_out = np.where(in_or_out == False)
pts_other = pts_new[indices_out]
pts_new = pts_new[indices_in]

# Generate KDTrees
pnt_tree_orig = spatial.cKDTree(pts_original)


# pnt_tree_new = spatial.cKDTree(pts_new)
#
# r = 50
# all_pairs = pnt_tree_new.query_ball_tree(pnt_tree_orig, r)
#
# i = 0
# max_pair = 0
# dat_new = np.zeros(np.shape(pts_new)[0])
# for pairs in all_pairs:
#     if len(pairs) < 3:
#         i += 1
#         continue
#     if len(pairs) > max_pair:
#         max_pair = len(pairs)
#
#     # Original coords and data
#     x_c_orig, y_c_orig, z_c_orig = pnt_tree_orig.data[pairs].T
#     dat_orig = data[pairs]
#
#     rbfi = Rbf(x_c_orig, y_c_orig, z_c_orig, dat_orig)
#
#     # New coords and data
#     pts_local = np.array((x_c_orig, y_c_orig, z_c_orig)).T
#     x_c_new, y_c_new, z_c_new = pnt_tree_new.data[i]
#     #print x_c_new, y_c_new
#
#     xi = np.array((x_c_new, y_c_new, z_c_new))
#
#     #dat_new[i] = griddata(pts_local, dat_orig, xi, method='nearest', fill_value=-5.0)
#     dat_new[i] = rbfi(x_c_new, y_c_new, z_c_new)
#     i += 1
#     if i % 100 == 0:
#         print(len(pairs))
#         print(i)


# Method based on nearest points:
_, pairs = pnt_tree_orig.query(pts_new, k=10)

i = 0
dat_new = np.zeros(np.shape(pts_new)[0])
for idx in pairs:
    x_c_orig, y_c_orig, z_c_orig = pts_original[idx].T
    dat_orig = data[idx]

    rbfi = Rbf(x_c_orig, y_c_orig, z_c_orig, dat_orig)
    x_c_new, y_c_new, z_c_new = pts_new[i]
    dat_new[i] = rbfi(x_c_new, y_c_new, z_c_new)

    i += 1

    if i % 100 == 0:
        print(len(pairs))
        print(i)



pts_all = np.append(pts_new, pts_other, axis=0)
dat_all = np.append(dat_new, np.zeros_like(pts_other[:,0]))

# Write to vtk
x, y, z = pts_all.T
elements = triangulate(x, y, z)
write_vtk(os.path.join('ses_check.vtk'), pts_all, elements, dat_all, 'ses3d')


# x, y, z = pts_all.T
# #dat_hull = np.ones_like(x)
# elements = triangulate(x, y, z)
# #print(np.shape(pts_hull))
#
# pts = np.array((x, y, z)).T
# write_vtk("ses3d_kd_tree.vtk", pts_all, elements, dat_all, 'hull')
