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

    def split_domains(self, pts):
        # Split array in two zones
        s20rts_dmn = pts[pts[:, 2] <= self.layers[0]]
        s20rts_dmn = s20rts_dmn[s20rts_dmn[:, 2] >= self.layers[-1]]
        above_dmn = pts[pts[:, 2] > self.layers[0]]
        below_dmn = pts[pts[:, 2] < self.layers[-1]]
        non_s20_dmn = np.append(above_dmn, below_dmn, axis=0)
        return s20rts_dmn, non_s20_dmn



    def eval_point_cloud_non_norm(self, c, l, r, rho, vpv, vsv, vsh):
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

        self.read()
        pts = np.array((c, l, r, rho, vpv, vsv, vsh)).T
        s20rts_dmn, non_s20_dmn = self.split_domains(pts)
        # Initialize arrays to store evaluated points in the correct order
        c = np.zeros(0)
        l = np.zeros(0)
        r = np.zeros(0)
        rho = np.zeros(0)
        vpv = np.zeros(0)
        vsv = np.zeros(0)
        vsh = np.zeros(0)

        # Only run when it exists
        if len(s20rts_dmn) > 0:
            for i in range(len(self.layers) - 1):
                upper_rad = self.layers[i]
                lower_rad = self.layers[i+1]

                # Extract chunk for interpolation
                if i == 0:
                    chunk = s20rts_dmn[s20rts_dmn[:, 2] <= upper_rad + np.finfo(float).eps]
                else:
                    chunk = s20rts_dmn[s20rts_dmn[:, 2] <= upper_rad]

                if i < len(self.layers) - 2:
                    chunk = chunk[chunk[:, 2] > lower_rad]
                else:
                    chunk = chunk[chunk[:, 2] >= lower_rad - np.finfo(float).eps]

                chunk_c, chunk_l, chunk_r, chunk_rho, chunk_vpv, chunk_vsv, chunk_vsh = chunk.T

                # Evaluate S20RTS at upper and lower end of chunk, use these to interpolate
                top_vals = self.eval(chunk_c, chunk_l, upper_rad, 'test')
                bottom_vals = self.eval(chunk_c, chunk_l, lower_rad, 'test')

                # Interpolate
                chunk_vals = self.linear_interpolation(bottom_vals, top_vals, lower_rad, upper_rad, chunk_r)

                # Compute vp perturbations
                R0 = 1.25
                R2891 = 3.0
                vp_slope = (R2891 - R0) / 2891.0
                rDep = vp_slope * (self.r_earth - chunk_r) + R0
                vp_val = chunk_vals / rDep

                # Add perturbations
                chunk_vpv *= (1 + vp_val)
                chunk_vsv *= (1 + chunk_vals)
                chunk_vsh *= (1 + chunk_vals)

                # Store perturbed values
                c = np.append(c, chunk_c)
                r = np.append(r, chunk_r)
                l = np.append(l, chunk_l)
                rho = np.append(rho, chunk_rho)
                vpv = np.append(vpv, chunk_vpv)
                vsv = np.append(vsv, chunk_vsv)
                vsh = np.append(vsh, chunk_vsh)

        s20rts_dmn = np.array((c, l, r, rho, vpv, vsv, vsh)).T

        pts = np.append(s20rts_dmn, non_s20_dmn, axis=0)

        return pts

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
