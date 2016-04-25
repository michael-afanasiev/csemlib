import datetime
import math
import os

import pandas as pd
import xarray as xarray

from .model import Model


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
        col = pd.read_csv(os.path.join(self.directory, 'block_x'), header=1,
                          dtype=float).values
        lon = pd.read_csv(os.path.join(self.directory, 'block_y'), header=1,
                          dtype=float).values
        rad = pd.read_csv(os.path.join(self.directory, 'block_z'), header=1,
                          dtype=float).values

        # Get centers of boxes.
        col = 0.5 * (col[1:] + col[:-1])
        lon = 0.5 * (lon[1:] + lon[:-1])
        rad = 0.5 * (rad[1:] + rad[:-1])

        # Read in parameters.
        for p in self.components:
            val = pd.read_csv(os.path.join(self.directory, p), header=1,
                              dtype=float).values
            val = val.reshape(len(col), len(lon), len(rad))
            self._data[p] = (('col', 'lon', 'rad'), val)
            if 'rho' in p:
                self._data[p].attrs['units'] = 'g/cm3'
            else:
                self._data[p].attrs['units'] = 'km/s'

        # Add coordinates.
        self._data.coords['col'] = col.T[0] * math.pi / 180.0
        self._data.coords['lon'] = lon.T[0] * math.pi / 180.0
        self._data.coords['rad'] = rad.T[0]

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
