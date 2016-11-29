import io
import os

import numpy as np
import scipy.interpolate as interp
import xarray

from csemlib.models.model import Model


class Topography(Model):
    """
    Class handling crustal models.
    """

    def __init__(self):

        super(Topography, self).__init__()
        self._data = xarray.Dataset()

        directory = os.path.split(os.path.split(__file__)[0])[0]
        self.directory = os.path.join(directory, 'data', 'topography')
        self.col = np.zeros(0)
        self.lon = np.zeros(0)
        self.topo = np.zeros(0)

    def data(self):
        pass

    def read(self):

        col = np.linspace(1, 179, 180*6+1)
        lon = np.linspace(0, 360, 360*6+1)

        for p in ['topo']:

            with io.open(os.path.join(self.directory, 'new_vals'), 'rt') as fh:
                val = np.asarray(fh.readlines(), dtype=float)
            # val_new = []
            # for i in range(len(val)):
            #     if (i + 1) % len(lon) == 0:
            #         continue
            #
            #     else:
            #         val_new.append(val[i])
            #
            #
            # val = np.array(val_new)
            # np.savetxt(os.path.join(self.directory, 'new_vals'), val_new)
            lon = np.linspace(1, 360, 360*6, endpoint=False)
            val = val.reshape(len(col), len(lon))
            self._data[p] = (('col', 'lon'), val)
            if p == 'topo':
                self._data[p].attrs['units'] = 'km'


        # Add coordinates.
        self._data.coords['col'] = np.radians(col)
        self._data.coords['lon'] = np.radians(lon)

        # Add units.
        self._data.coords['col'].attrs['units'] = 'radians'
        self._data.coords['lon'].attrs['units'] = 'radians'


    def write(self):
        pass

    def eval(self, x, y, z=0, param=None, crust_smooth_factor=1e5):

        # This is a heuristic.
        #crust_smooth_factor = 1e5

        # Create smoother object.
        lut = interp.RectSphereBivariateSpline(self._data.coords['col'],
                                               self._data.coords['lon'],
                                               self._data[param])

        # Because the colatitude array is reversed, we must also reverse the request.
        x = np.pi - x

        # Because we use a definition of longitude ranging between -pi and pi
        # convert negative longitudes to positive longitudes, RectSphereBivariateSpline requires this
        # y[y < 0] = np.pi - y[y < 0]
        return lut.ev(x, y)


topo = Topography()
topo.read()
# print(np.size(topo._data['topo']))
print(topo.eval(0.0,1, param='topo'))
