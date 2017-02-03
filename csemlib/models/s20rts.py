import io
import os

import numpy as np
from scipy.special import sph_harm

from.model import Model


class S20rts(Model):
    """
    Class handling S20rts evaluations.
    """

    def data(self):
        pass

    def __init__(self):
        super(S20rts, self).__init__()
        directory, _ = os.path.split(os.path.split(__file__)[0])
        self.directory = os.path.join(directory, 'data', 's20rts')
        self.layers = np.array([6346.63, 6296.63, 6241.64, 6181.14, 6114.57, 6041.34, 5960.79,
                                5872.18, 5774.69, 5667.44, 5549.46, 5419.68, 5276.89, 5119.82,
                                4947.02, 4756.93, 4547.81, 4317.74, 4064.66, 3786.25, 3479.96])
        self.r_earth = 6371.0

    def read(self):
        coeff_file = os.path.join(self.directory, 'S20RTS.dat')
        with io.open(coeff_file, 'rt') as fh:
            coeffs = np.asarray(fh.read().split(), dtype=float)

        tot = 0
        for s in range(21):
            setattr(self, 'l%d' % s, [])
            for i in range(21):
                c = []
                for j in range(2 * i + 1):
                    c.append(coeffs[tot])
                    tot += 1
                getattr(self, 'l%d' % s).append(np.array(c))

    def write(self):
        pass

    def eval(self, c, l, rad):
        """
        This returns the perturbation as defined in s20rts. Only one rad can be handled at a time, while
        c and l can be given in the form of 1D arrays
        :param c: colatitude,
        :param l: longitude,
        :param rad: distance from core in km
        :param param: param to be returned - currently not used
        :return vals
        """

        if rad not in self.layers:
            raise ValueError('Requested layer not defined in s20rts, use interpolation function')
        idx = self.find_layer_idx(rad)

        vals = np.zeros_like(c)
        for n in range(20):
            imag, real = 1, 0
            for m in range(2 * n + 1):
                if m == 0 or (m % 2):
                    vals += getattr(self, 'l%d' % idx)[n][m] * np.real(sph_harm(real, n, l, c))
                    real += 1
                else:
                    vals += getattr(self, 'l%d' % idx)[n][m] * np.imag(sph_harm(imag, n, l, c))
                    imag += 1

        return vals


    def find_layer_idx(self, rad):
        """
        Return the index of the s20rts layer for the requested depth
        :param rad: distance from core in km
        :return layer index:
        """
        if rad not in self.layers:
            raise ValueError('Requested layer not defined in s20rts, use interpolation function')

        layer_idx, _ = min(enumerate(self.layers), key=lambda x: abs(x[1] - rad))

        return layer_idx

    def linear_interpolation(self, bottom_vals, top_vals, bottom_rad, top_rad, rads):
        """
        Returns the linear interpolated value
        :param bottom_vals: perturbation at layer below point
        :param top_vals: perturbation at layer above point
        :param bottom_rad: radius of the layer below
        :param top_rad: radius of the layer above
        :param rads: radius of the to be interpolated value
        :return: vals: interpolated results
        """

        vals = (top_vals - bottom_vals)/(top_rad-bottom_rad) *\
               (rads - bottom_rad) + bottom_vals

        return vals

    def split_domains_griddata(self, GridData):
        """
        This splits an array of pts of all values into a
        :param pts:
        :return:
        """

        s20rts_dmn = GridData.copy()
        s20rts_dmn.df = s20rts_dmn.df[s20rts_dmn.df['r'] <= self.layers[0]]
        s20rts_dmn.df = s20rts_dmn.df[s20rts_dmn.df['r'] >= self.layers[-1]]

        return s20rts_dmn


    def eval_point_cloud_griddata(self, GridData):
        """
        This returns the linearly interpolated perturbations of s20rts. Careful only points that fall inside
        of the domain of s20rts are returned.
        :param c: colatitude
        :param l: longitude
        :param r: distance from core in km
        :param rho: param to be returned - currently not used
        :param vpv: param to be returned - currently not used
        :param vsv: param to be returned - currently not used
        :param vsh: param to be returned - currently not used
        :return c, l, r, rho, vpv, vsv, vsh
        """
        print('Evaluating S20RTS')
        self.read()
        s20rts_dmn = self.split_domains_griddata(GridData)

        # Only run when it exists
        if len(s20rts_dmn) > 0:
            for i in range(len(self.layers) - 1):
                upper_rad = self.layers[i]
                lower_rad = self.layers[i+1]

                # Extract chunk for interpolation
                chunk = s20rts_dmn.df[s20rts_dmn.df['r'] <= upper_rad]

                if i < len(self.layers) - 2:
                    chunk = chunk[chunk['r'] > lower_rad]
                else:
                    chunk = chunk[chunk['r'] >= lower_rad]

                # Evaluate S20RTS at upper and lower end of chunk, use these to interpolate
                top_vals = self.eval(chunk['c'], chunk['l'], upper_rad)
                bottom_vals = self.eval(chunk['c'], chunk['l'], lower_rad)

                # Interpolate
                chunk_vals = self.linear_interpolation(bottom_vals, top_vals, lower_rad, upper_rad, chunk['r'])

                # Compute vp perturbations
                R0 = 1.25
                R2891 = 3.0
                vp_slope = (R2891 - R0) / 2891.0
                rDep = vp_slope * (self.r_earth - chunk['r']) + R0
                vp_val = chunk_vals / rDep

                # Add perturbations
                if 'vpv' in chunk.columns:
                    chunk['vpv'] *= (1 + vp_val)

                if 'vph' in chunk.columns:
                    chunk['vph'] *= (1 + vp_val)

                if 'vp' in chunk.columns:
                    chunk['vp'] *= (1 + vp_val)

                if 'vsv' in chunk.columns:
                    chunk['vsv'] *= (1 + chunk_vals)

                if 'vsh' in chunk.columns:
                    chunk['vsh'] *= (1 + chunk_vals)

                s20rts_dmn.df.update(chunk)

        GridData.df.update(s20rts_dmn.df)

        return GridData
