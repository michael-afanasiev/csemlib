import io
import os

import numpy as np
import scipy.interpolate as interp
import xarray

from .model import Model
from csemlib.models.topography import Topography



class Crust(Model):
    """
    Class handling crustal models.
    """

    def __init__(self):

        super(Crust, self).__init__()
        self._data = xarray.Dataset()

        directory = os.path.split(os.path.split(__file__)[0])[0]
        self.directory = os.path.join(directory, 'data', 'crust')
        self._data = xarray.Dataset()

    def data(self):
        pass

    def read(self):

        with io.open(os.path.join(self.directory, 'crust_x'), 'rt') as fh:
            col = np.asarray(fh.readlines(), dtype=float)

        with io.open(os.path.join(self.directory, 'crust_y'), 'rt') as fh:
            lon = np.asarray(fh.readlines(), dtype=float)

        for p in ['crust_dep', 'crust_vs']:

            with io.open(os.path.join(self.directory, p), 'rt') as fh:
                val = np.asarray(fh.readlines(), dtype=float)
            val = val.reshape(len(col), len(lon))
            self._data[p] = (('col', 'lon'), val)
            if p == 'crust_dep':
                self._data[p].attrs['units'] = 'km'
            elif p == 'crust_vs':
                self._data[p].attrs['units'] = 'km/s'

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
        lut = interp.RectSphereBivariateSpline(self._data.coords['col'][::-1],
                                               self._data.coords['lon'],
                                               self._data[param],
                                               s=crust_smooth_factor)

        # Because the colatitude array is reversed, we must also reverse the request.
        x = np.pi - x

        # Convert longitudes to coordinate system of the crustal model
        lon = np.copy(y)
        lon[lon < 0] = 2 * np.pi + lon[lon < 0]
        return lut.ev(x, lon)


    def eval_point_cloud(self, c, l, r, rho, vpv, vsv, vsh):

        pts = np.array((c, l, r, rho, vpv, vsv, vsh)).T
        r_earth = 6371.0

        # Split into crustal and non crustal zone
        cst_zone = pts[pts[:, 2] >= (r_earth - 100.0)]
        non_cst_zone = pts[pts[:, 2] < (r_earth - 100.0)]

        # Compute crustal depths and vs for crustal zone coordinates
        self.read()

        # This guy was changing the coordinates which somehow made it work before, with that fixed it does not work anymore
        crust_dep = self.eval(cst_zone[:, 0], cst_zone[:, 1], param='crust_dep', crust_smooth_factor=1e1)
        crust_vs = self.eval(cst_zone[:, 0], cst_zone[:, 1], param='crust_vs', crust_smooth_factor=0)

        # Get Topography
        top = Topography()
        top.read()
        topo = top.eval(cst_zone[:, 0], cst_zone[:, 1], param='topo')

        # Increase crustal depth for regions with topography
        crust_dep[topo >= 0] = crust_dep[topo >= 0] + topo[topo >= 0]

        # Add crust
        cst_zone[:, 3], cst_zone[:, 4], cst_zone[:, 5], cst_zone[:, 6] = \
            add_crust_all_params_topo(cst_zone[:, 2], crust_dep, crust_vs,
                                topo, cst_zone[:, 3], cst_zone[:, 4], cst_zone[:, 5], cst_zone[:, 6])
        # Append crustal and non crustal zone back together
        pts = np.append(cst_zone, non_cst_zone, axis=0)

        return pts

    def eval_point_cloud_grid_data(self, GridData):
            r_earth = 6371.0

            # Split into crustal and non crustal zone
            cst_zone = GridData.df[GridData.df['r'] >= (r_earth - 100.0)]

            # Compute crustal depths and vs for crustal zone coordinates
            self.read()

            # This guy was changing the coordinates which somehow made it work before, with that fixed it does not work anymore
            crust_dep = self.eval(cst_zone['c'], cst_zone['l'], param='crust_dep', crust_smooth_factor=1e1)
            crust_vs = self.eval(cst_zone['c'], cst_zone['l'], param='crust_vs', crust_smooth_factor=0)

            # Get Topography
            top = Topography()
            top.read()
            topo = top.eval(cst_zone['c'], cst_zone['l'], param='topo')

            # Increase crustal depth for regions with topography
            crust_dep[topo >= 0] = crust_dep[topo >= 0] + topo[topo >= 0]

            # Add crust
            cst_zone = add_crust_all_params_topo_griddata(cst_zone, crust_dep, crust_vs, topo)
            # Append crustal and non crustal zone back together
            GridData.df.update(cst_zone)

            return GridData


def add_crust_all_params_topo(r, crust_dep, crust_vs, topo, rho, vpv, vsv, vsh):
    r_earth = 6371.0
    for i in range(len(r)):
        if r[i] > (r_earth - crust_dep[i]):
            # Do something with param here
            vsv[i] = crust_vs[i]
            vsh[i] = crust_vs[i]

            # Continental crust
            if topo[i] >= 0:
                vpv[i] = 1.5399 * crust_vs[i] + 0.840
                rho[i] = 0.2277 * crust_vs[i] + 2.016

            # Oceanic crust
            if topo[i] < 0:
                vpv[i] = 1.5865 * crust_vs[i] + 0.844
                rho[i] = 0.2547 * crust_vs[i] + 1.979
        else:
            continue

    return rho, vpv, vsv, vsh



def add_crust_all_params_topo_griddata(cst_zone, crust_dep, crust_vs, topo):
    r_earth = 6371.0
    for i in range(len(cst_zone['r'])):
        if cst_zone['r'].values[i] > (r_earth - crust_dep[i]):
            # Do something with param here
            if 'vsv' in cst_zone.columns:
                cst_zone['vsv'].values[i] = crust_vs[i]
            if 'vsh' in cst_zone.columns:
                cst_zone['vsh'].values[i] = crust_vs[i]

            # Continental crust
            if topo[i] >= 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = 1.5399 * crust_vs[i] + 0.840
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = 1.5399 * crust_vs[i] + 0.840
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = 0.2277 * crust_vs[i] + 2.016

            # Oceanic crust
            if topo[i] < 0:
                if 'vpv' in cst_zone.columns:
                    cst_zone['vpv'].values[i] = 1.5865 * crust_vs[i] + 0.844
                if 'vph' in cst_zone.columns:
                    cst_zone['vph'].values[i] = 1.5865 * crust_vs[i] + 0.844
                if 'rho' in cst_zone.columns:
                    cst_zone['rho'].values[i] = 0.2547 * crust_vs[i] + 1.979

        else:
            continue

    return cst_zone
