import os

import numpy as np
import scipy.interpolate as interp
import xarray

from .model import Model

class Topography(Model):
    """
    Class handling topography models.
    """

    def __init__(self):

        super(Topography, self).__init__()
        self._data = xarray.Dataset()

        directory = os.path.split(os.path.split(__file__)[0])[0]
        self.directory = os.path.join(directory, 'data', 'topography')

    def data(self):
        pass

    def read(self):

        # initial_reading = False
        # if initial_reading:
        #     # Initial Reading
        #     vals = np.genfromtxt(os.path.join(self.directory, '10MinuteTopoGrid.txt'), delimiter=',')
        #     _, _, topo = vals.T
        #
        #     initial_lon = np.linspace(-180, 180, 360 * 6 + 1)
        #     initial_col = np.linspace(-90, 90, 180 * 6 + 1)
        #
        #     topo = np.array(topo)
        #     topo_reshaped = topo.reshape(len(initial_col), len(initial_lon))
        #
        #     # Resample such that there are no points at the poles
        #     topo_resampled = topo_reshaped[1::2, 1::2]
        #     topo_1d = topo_resampled.reshape(np.size(topo_resampled))
        #     np.savetxt(os.path.join(self.directory, 'topo_resampled'), topo_1d, fmt='%.0f')

        val = np.genfromtxt(os.path.join(self.directory, 'topo_resampled'))
        # new sampling:
        start = 1.0/6.0
        col = np.linspace(start, 180 - start, 540)
        lon = np.linspace(start, 360 - start, 1080)
        # Reshape
        val = val.reshape(len(col), len(lon))

        # Convert to km
        val /= 1000.0
        self._data['topo'] = (('col', 'lon'), val)
        self._data['topo'].attrs['units'] = 'km'

        # Add coordinates.
        self._data.coords['col'] = np.radians(col)
        self._data.coords['lon'] = np.radians(lon)

        # Add units.
        self._data.coords['col'].attrs['units'] = 'radians'
        self._data.coords['lon'].attrs['units'] = 'radians'


    def write(self):
        pass

    def eval(self, x, y, z=0, param=None, topo_smooth_factor=0):

        # This is a heuristic.
        #topo_smooth_factor = 1e2

        # Create smoother object.
        lut = interp.RectSphereBivariateSpline(self._data.coords['col'],
                                               self._data.coords['lon'],
                                               self._data[param],
                                               s=topo_smooth_factor, pole_values=[-4.228, -0.056])

        # Convert to coordinate system used for topography 0-2pi instead of -pi-pi
        y = y + np.pi
        return lut.ev(x, y)
