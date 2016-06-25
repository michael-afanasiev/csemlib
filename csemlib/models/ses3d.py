import datetime
import io
import os

import numpy as np
import xarray

from .model import Model, shade, triangulate, interpolate


class Ses3d(Model):
    """
    Class handling file-IO for a model in SES3D format.
    """

    def __init__(self, name, directory,
                 components=[], doi=None):
        super(Ses3d, self).__init__()
        self._data = xarray.Dataset()
        self.directory = directory
        self.components = components
        if doi:
            self.doi = doi
        else:
            self.doi = 'None'

    @property
    def data(self):
        return self._data

    def read(self):

        files = set(os.listdir(self.directory))
        if self.components:
            if not set(self.components).issubset(files):
                raise IOError(
                    'Model directory does not have all components ' +
                    ', '.join(self.components))

        # Read values.
        with io.open(os.path.join(self.directory, 'block_x'), 'rt') as fh:
            col = np.asarray(fh.readlines()[2:], dtype=float)
        with io.open(os.path.join(self.directory, 'block_y'), 'rt') as fh:
            lon = np.asarray(fh.readlines()[2:], dtype=float)
        with io.open(os.path.join(self.directory, 'block_z'), 'rt') as fh:
            rad = np.asarray(fh.readlines()[2:], dtype=float)

        # Get centers of boxes.
        col = 0.5 * (col[1:] + col[:-1])
        lon = 0.5 * (lon[1:] + lon[:-1])
        rad = 0.5 * (rad[1:] + rad[:-1])

        # Read in parameters.
        for p in self.components:
            with io.open(os.path.join(self.directory, p), 'rt') as fh:
                val = np.asarray(fh.readlines()[2:], dtype=float)
            val = val.reshape(col.size, lon.size, rad.size)
            self._data[p] = (('col', 'lon', 'rad'), val)
            if 'rho' in p:
                self._data[p].attrs['units'] = 'g/cm3'
            else:
                self._data[p].attrs['units'] = 'km/s'

        # Add coordinates.
        self._data.coords['col'] = np.radians(col)
        self._data.coords['lon'] = np.radians(lon)
        self._data.coords['rad'] = rad

        # Add units.
        self._data.coords['col'].attrs['units'] = 'radians'
        self._data.coords['lon'].attrs['units'] = 'radians'
        self._data.coords['rad'].attrs['units'] = 'km'

        # Add Ses3d attributes.
        self._data.attrs['solver'] = 'ses3d'
        self._data.attrs['coordinate_system'] = 'spherical'
        self._data.attrs['date'] = datetime.datetime.now().__str__()
        self._data.attrs['doi'] = self.doi

    def write(self):
        print('Writing')

    def eval(self, x, y, z, param=None):
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
            self._data.coords['col'].values,
            self._data.coords['lon'].values,
            self._data.coords['rad'].values)
        cols = cols.ravel()
        lons = lons.ravel()
        rads = rads.ravel()

        # Generate tetrahedra.
        elements = triangulate(cols, lons, rads)

        # Get interpolating functions.
        indices, barycentric_coordinates = shade(x, y, z, cols, lons, rads, elements)
        interp_param = []
        for i, p in enumerate(param):
            interp_param.append(np.array(
                interpolate(indices, barycentric_coordinates, self.data[p].values.ravel()), dtype=np.float64))

        return np.array(interp_param).T
