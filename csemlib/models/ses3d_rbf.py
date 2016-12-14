import datetime
import io
import os

import numpy as np
import xarray


from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.model import Model, shade, triangulate, interpolate, write_vtk
from csemlib.utils import sph2cart, rotate, cart2sph


# from .model import Model, shade, triangulate, interpolate
# from ..utils import sph2cart, rotate


def _read_multi_region_file(data):
    regions = []
    num_region, region_start = None, None
    num_regions = int(data[0])
    for i in range(num_regions):
        region_start = int(region_start + num_region + 1) if region_start else int(2)
        num_region = int(data[region_start - 1]) if num_region else int(data[1])
        regions.append(data[region_start:region_start + num_region])

    return regions


def _setup_rot_matrix(angle, x, y, z):
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


class Ses3d(Model):
    """
    Class handling file-IO for a model in SES3D format.
    """

    def __init__(self, name, directory, components=[],
                 rotation_vector=None, rotation_angle=None, doi=None):
        super(Ses3d, self).__init__()
        self.rot_angle = rotation_angle
        self.rot_mat = rotation_vector
        self._data = []
        self.directory = directory
        self.components = components
        if doi:
            self.doi = doi
        else:
            self.doi = 'None'

    @property
    def data(self, region=0):
        return self._data[region]

    def read(self):

        files = set(os.listdir(self.directory))
        if self.components:
            if not set(self.components).issubset(files):
                raise IOError(
                    'Model directory does not have all components ' +
                    ', '.join(self.components))

        # Read values.
        with io.open(os.path.join(self.directory, 'block_x'), 'rt') as fh:
            data = np.asarray(fh.readlines(), dtype=float)
            col_regions = _read_multi_region_file(data)
        with io.open(os.path.join(self.directory, 'block_y'), 'rt') as fh:
            data = np.asarray(fh.readlines(), dtype=float)
            lon_regions = _read_multi_region_file(data)
        with io.open(os.path.join(self.directory, 'block_z'), 'rt') as fh:
            data = np.asanyarray(fh.readlines(), dtype=float)
            rad_regions = _read_multi_region_file(data)

        # Get centers of boxes.
        for i, _ in enumerate(col_regions):
            col_regions[i] = 0.5 * (col_regions[i][1:] + col_regions[i][:-1])
            lon_regions[i] = 0.5 * (lon_regions[i][1:] + lon_regions[i][:-1])
            rad_regions[i] = 0.5 * (rad_regions[i][1:] + rad_regions[i][:-1])

        # Read in parameters.
        for p in self.components:
            with io.open(os.path.join(self.directory, p), 'rt') as fh:
                data = np.asarray(fh.readlines(), dtype=float)
                val_regions = _read_multi_region_file(data)

            for i, _ in enumerate(val_regions):
                val_regions[i] = val_regions[i].reshape((len(col_regions[i]), len(lon_regions[i]),
                                                         len(rad_regions[i])))
                if not self._data:
                    self._data = [xarray.Dataset() for i in range(len(val_regions))]

                self._data[i][p] = (('col', 'lon', 'rad'), val_regions[i])
                if 'rho' in p:
                    self._data[i][p].attrs['units'] = 'g/cm3'
                else:
                    self._data[i][p].attrs['units'] = 'km/s'

        # Add coordinates.
        for i, _ in enumerate(val_regions):
            s_col, s_lon, s_rad = len(col_regions[i]), len(lon_regions[i]), len(rad_regions[i])
            self._data[i].coords['col'] = np.radians(col_regions[i])
            self._data[i].coords['lon'] = np.radians(lon_regions[i])
            self._data[i].coords['rad'] = rad_regions[i]

            cols, lons, rads = np.meshgrid(self._data[i].coords['col'].values,
                                           self._data[i].coords['lon'].values,
                                           self._data[i].coords['rad'].values)

            # Cartesian coordinates and rotation.
            x, y, z = sph2cart(cols.ravel(), lons.ravel(), rads.ravel())
            if self.rot_mat:
                if len(self.rot_mat) is not 3:
                    raise ValueError("Rotation matrix must be a 3-vector.")
                self.rot_mat = _setup_rot_matrix(np.radians(self.rot_angle), *self.rot_mat)
                x, y, z = rotate(x, y, z, self.rot_mat)

            self._data[i]['x'] = (('col', 'lon', 'rad'), x.reshape((s_col, s_lon, s_rad)))
            self._data[i]['y'] = (('col', 'lon', 'rad'), y.reshape((s_col, s_lon, s_rad)))
            self._data[i]['z'] = (('col', 'lon', 'rad'), z.reshape((s_col, s_lon, s_rad)))

            # Add units.
            self._data[i].coords['col'].attrs['units'] = 'radians'
            self._data[i].coords['lon'].attrs['units'] = 'radians'
            self._data[i].coords['rad'].attrs['units'] = 'km'

            # Add Ses3d attributes.
            self._data[i].attrs['solver'] = 'ses3d'
            self._data[i].attrs['coordinate_system'] = 'spherical'
            self._data[i].attrs['date'] = datetime.datetime.now().__str__()
            self._data[i].attrs['doi'] = self.doi

    def write(self, directory):

        for block, comp in zip(['block_x', 'block_y', 'block_z'], ['col', 'lon', 'rad']):
            with io.open(os.path.join(directory, block), 'w') as fh:
                fh.write("1\n")
                fh.write(str(len(self.data.coords[comp].values)) + "\n")
                if block in ['block_x', 'block_y']:
                    fh.write('\n'.join([str(num) for num in np.degrees(self.data.coords[comp].values)]))
                else:
                    fh.write('\n'.join([str(num) for num in self.data.coords[comp].values]))

        for par in self.components:
            with io.open(os.path.join(directory, par), 'w') as fh:
                fh.write("1\n")
                fh.write(str(len(self.data[par].values.ravel())) + "\n")
                fh.write('\n'.join([str(num) for num in self.data[par].values.ravel()]))

    def eval(self, x, y, z, param=None, region=0):
        """
        Return the interpolated parameter at a spatial location.

        For Ses3D models, we rely on Delauny triangulation and barycentric
        interpolation to determine model values away from grid points. This
        function will first use meshpy to build up the triangulation. Then,
        a Kd-tree will be created to locate the closest tetrahedral nodes.
        Finally, enclosing tetrahedra are found by checking the barycentric
        coordinates, and linear interpolation is performed over the enclosing
        simplex.
        :param x: X coordinate.
        :param y: Y coordinate.
        :param z: Z coordinate.
        :param param: Param to interpolate.
        :return: Interpolated param at (x, y, z).
        """

        # Pack up the points.
        cols, lons, rads = np.meshgrid(
            self._data[region].coords['col'].values,
            self._data[region].coords['lon'].values,
            self._data[region].coords['rad'].values)
        cols = cols.ravel()
        lons = lons.ravel()
        rads = rads.ravel()

        # Generate tetrahedra. Currently this in spherical coordinates, as we need a convex hull.
        elements = triangulate(cols, lons, rads)

        # Get interpolating functions. Map cartesian coordinates for the interpolation.
        interp_param = []
        indices, barycentric_coordinates = shade(x, y, z, self.data['x'].values.ravel(),
                                                 self.data['y'].values.ravel(), self.data['z'].values.ravel(),
                                                 elements)
        for i, p in enumerate(param):
            interp_param.append(np.array(
                interpolate(indices, barycentric_coordinates, self.data[p].values.ravel()), dtype=np.float64))

        return np.array(interp_param).T


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


import numpy as np
import _pickle as pickle
import scipy.spatial as spatial
from scipy.interpolate import Rbf
from scipy.spatial.qhull import ConvexHull
from scipy.interpolate import griddata

# Get original data (this seems to work fine)
mod = Ses3d('japan', os.path.join('/home/sed/CSEM/csemlib/tests/test_data', 'japan'),
                components=['drho', 'dvsv', 'dvsh', 'dvp'])
mod.read()

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
radii = np.linspace(6250.0, 0.0, 3)
resolution = np.ones_like(radii) * (6350.0 / 10)
fib_grid.set_global_sphere(radii, resolution)
# refinement region coarse
c_min = np.radians(10)
c_max = np.radians(90)
l_min = np.radians(80)
l_max = np.radians(170)
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
