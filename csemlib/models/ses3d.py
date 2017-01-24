import datetime
import io
import os
import yaml

import numpy as np
import xarray

from .model import Model, shade, triangulate, interpolate
from ..utils import sph2cart, rotate


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

    def __init__(self, name, directory, components=[], doi=None):
        super(Ses3d, self).__init__()
        self.rot_mat = None
        self._disc = []
        self._data = []
        self.directory = directory
        self.components = components
        if doi:
            self.doi = doi
        else:
            self.doi = 'None'

        with io.open(os.path.join(self.directory, 'modelinfo.yml'), 'rt') as fh:
            try:
                self.model_info = yaml.load(fh)
            except yaml.YAMLError as exc:
                print(exc)

        self.geometry = self.model_info['geometry']
        self.rot_vec = np.array([self.geometry['rot_x'], self.geometry['rot_y'], self.geometry['rot_z']])
        self.rot_angle = self.geometry['rot_angle']

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
            discretizations = {
                    'col': (col_regions[i][1] - col_regions[i][0]) / 2.0,
                    'lon': (lon_regions[i][1] - lon_regions[i][0]) / 2.0,
                    'rad': (rad_regions[i][1] - rad_regions[i][0]) / 2.0}
            self._disc.append(discretizations)
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
                                                         len(rad_regions[i])), order='C')
                if not self._data:
                    self._data = [xarray.Dataset() for j in range(len(val_regions))]
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

            lons, cols, rads = np.meshgrid(self._data[i].coords['lon'].values,
                                           self._data[i].coords['col'].values,
                                           self._data[i].coords['rad'].values)

            # Cartesian coordinates and rotation.
            x, y, z = sph2cart(cols.ravel(), lons.ravel(), rads.ravel())
            if self.model_info['geometry']['rotation']:
                if len(self.rot_vec) is not 3:
                    raise ValueError("Rotation matrix must be a 3-vector.")
                self.rot_mat = _setup_rot_matrix(np.radians(self.geometry['rot_angle']), *self.rot_vec)
                x, y, z = rotate(x, y, z, self.rot_mat)

            self._data[i]['x'] = (('col', 'lon', 'rad'), x.reshape((s_col, s_lon, s_rad), order='C'))
            self._data[i]['y'] = (('col', 'lon', 'rad'), y.reshape((s_col, s_lon, s_rad), order='C'))
            self._data[i]['z'] = (('col', 'lon', 'rad'), z.reshape((s_col, s_lon, s_rad), order='C'))

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
            with io.open(os.path.join(directory, block), 'wt') as fh:
                fh.write(str(len(self._data)) + u"\n")
                for region in range(len(self._data)):
                    fh.write(str(len(self._data[region].coords[comp].values) +
                        1) + u"\n")
                    if block in ['block_x', 'block_y']:
                        fh.write(u'\n'.join([str(num) for num in
                            np.degrees(self._data[region].coords[comp].values) -
                                self._disc[region][comp]]))
                        fh.write(u"\n" + str(np.degrees(self._data[region].coords[comp].values[-1])
                            + self._disc[region][comp]))
                    else:
                        fh.write(u'\n'.join([str(num) for num in
                            self._data[region].coords[comp].values -
                            self._disc[region][comp]]))
                        fh.write(u"\n" +
                                str(self._data[region].coords[comp].values[-1]
                                + self._disc[region][comp]))
                    fh.write(u"\n")

        for par in self.components:
            with io.open(os.path.join(directory, par), 'wt') as fh:
                fh.write(str(len(self._data)) + u"\n")
                for region in range(len(self._data)):
                    fh.write(str(len(self._data[region][par].values.ravel())) +
                            u"\n")
                    fh.write(u'\n'.join([str(num) for num in self._data[region][par].values.ravel()]))
                    fh.write(u"\n")

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
        indices, barycentric_coordinates = shade(x, y, z, self._data[region]['x'].values.ravel(),
                                                 self._data[region]['y'].values.ravel(),
                                                 self._data[region]['z'].values.ravel(),
                                                 elements)
        for i, p in enumerate(param):
            interp_param.append(np.array(
                interpolate(indices, barycentric_coordinates, self._data[region][p].values.ravel()), dtype=np.float64))

        return np.array(interp_param).T


    def eval_point_cloud_griddata(self, GridData):
        # Read model
        self.components = GridData.components
        self.read()

        # Get dmn
        # Turn off for europe to test
        ses3d_dmn = self.extract_ses3d_dmn(GridData)

        # Interpolate
        interp = self.eval(ses3d_dmn.df['x'], ses3d_dmn.df['y'], ses3d_dmn.df['z'], self.components)

        for i, p in enumerate(self.components):
            if self.model_info['component_type'] == 'perturbation':
                ses3d_dmn.df[p] += interp[:, i]
            elif self.model_info['component_type'] == 'absolute':
                ses3d_dmn.df[p] = interp[:, i]

        GridData.df.update(ses3d_dmn.df)

        return GridData

    def extract_ses3d_dmn(self, GridData):
        geometry = self.model_info['geometry']
        ses3d_dmn = GridData.copy()

        # Rotate
        if geometry['rotation'] is True:
            ses3d_dmn.rotate(-np.radians(geometry['rot_angle']), geometry['rot_x'],
                             geometry['rot_y'], geometry['rot_z'])
        # Extract
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['c'] >= np.deg2rad(geometry['cmin'])]
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['c'] <= np.deg2rad(geometry['cmax'])]
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['l'] >= np.deg2rad(geometry['lmin'])]
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['l'] <= np.deg2rad(geometry['lmax'])]
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['r'] >= geometry['rmin']]
        ses3d_dmn.df = ses3d_dmn.df[ses3d_dmn.df['r'] <= geometry['rmax']]

        # Rotate Back
        if geometry['rotation'] is True:
            ses3d_dmn.rotate(np.radians(geometry['rot_angle']), geometry['rot_x'],
                             geometry['rot_y'], geometry['rot_z'])

        return ses3d_dmn
