import io
import os

import numpy as np
from scipy.special import sph_harm

from csemlib.background.skeleton import multiple_fibonacci_spheres
from csemlib.models.model import Model, triangulate, write_vtk
from csemlib.utils import cart2sph, sph2cart


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
                                5872.18, 5774.69, 5667.44, 5549.46, 5419.68,5276.89, 5119.82,
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

    def eval(self, c, l, rad, param):
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

    def eval_point_cloud(self, c, l, r, param):
        """
        This returns the linearly interpolated perturbations of s20rts. Careful only points that fall inside
        of the domain of s20rts are returned.
        :param c: colatitude
        :param l: longitude
        :param r: normalised distance from core in km
        :param param: param to be returned - currently not used
        :return c, l, r, vals
        """
        pts = np.array((c, l, r)).T
        s20_lay_norm = self.layers / self.r_earth

        # Sorted array, probably not necessary anymore
        pts_sorted = np.asarray(sorted(pts, key=lambda pts_entry: pts_entry[2], reverse=True))

        # Initialize arrays to store evaluated points in the correct order
        vals = np.zeros(0)
        c = np.zeros(0)
        l = np.zeros(0)
        r = np.zeros(0)

        for i in range(len(self.layers) - 1):
            upper_rad_norm = s20_lay_norm[i]
            lower_rad_norm = s20_lay_norm[i+1]
            upper_rad = self.layers[i]
            lower_rad = self.layers[i+1]

            # Extract chunk for interpolation
            # Discard everything above chunk
            if i == 0:
                chunk = pts_sorted[pts_sorted[:, 2] <= upper_rad_norm + np.finfo(float).eps]
            else:
                chunk = pts_sorted[pts_sorted[:, 2] <= upper_rad_norm]

            # Discard everything below chunk
            if i < len(self.layers) - 2:
                chunk = chunk[chunk[:, 2] > lower_rad_norm]
            else:
                chunk = chunk[chunk[:, 2] >= lower_rad_norm - np.finfo(float).eps]

            chunk_c, chunk_l, chunk_r = chunk.T

            # Evaluate S20RTS at upper and lower end of chunk, use these to interpolate
            top_vals = self.eval(chunk_c, chunk_l, upper_rad, 'test')
            bottom_vals = self.eval(chunk_c, chunk_l, lower_rad, 'test')

            chunk_vals = self.linear_interpolation(bottom_vals, top_vals, lower_rad_norm, upper_rad_norm, chunk_r)
            vals = np.append(vals, chunk_vals)
            c = np.append(c, chunk_c)
            r = np.append(r, chunk_r)
            l = np.append(l, chunk_l)

        return c, l, r, vals

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
