import math
import numpy as np

from csemlib.utils import cart2sph, sph2cart


class FibonacciGrid:
    """
    A class handling the generation of a Fibonacci grid.
    """


    def __init__(self):
        self._x = np.zeros(0)
        self._y = np.zeros(0)
        self._z = np.zeros(0)
        self.r_earth = 6371.0
        self.has_zero_point = False

    def set_global_sphere(self, radii, resolution):
        """
        Define a global Fibonacci grid
        :param radii: array specifiying all the radius where a layer should be added
        :param resolution: array specifying the resolution of these layers in km
        :return:
        """
        for i in range(len(radii)):
            if radii[i] <= np.finfo(float).eps:
                if self.has_zero_point:
                    continue
                else:
                    self.has_zero_point = True
                    self._x = np.append(self._x, np.zeros(1))
                    self._y = np.append(self._y, np.zeros(1))
                    self._z = np.append(self._z, np.zeros(1))
                    continue

            surf_area_sphere = (4 * np.pi * radii[i] ** 2)
            n_samples = int(surf_area_sphere / (resolution[i] ** 2)) + 1
            x, y, z = self.adaptable_fibonacci_sphere(n_samples)

            self._x = np.append(self._x, x * radii[i])
            self._y = np.append(self._y, y * radii[i])
            self._z = np.append(self._z, z * radii[i])


    def add_refinement_region(self, c_min, c_max, l_min, l_max, radii, resolution):
        """
        This function returns a refinement region
        :param c_min: minimum colatitude in radians
        :param c_max: maximum colatitude in radians
        :param l_min: minimum longitude in radians
        :param l_max: maximum longitude in radians
        :param radii: array specifiying all the radius where a layer should be added
        :param resolution: array specifying the resolution of these layers in km
        :return:
        """

        #self.discard_region(c_min, c_max, l_min, l_max, np.min(radii), np.max(radii))

        for i in range(len(radii)):
            if radii[i] <= np.finfo(float).eps:
                if self.has_zero_point:
                    continue
                else:
                    self.has_zero_point = True
                    self._x = np.append(self._x, np.zeros(1))
                    self._y = np.append(self._y, np.zeros(1))
                    self._z = np.append(self._z, np.zeros(1))
                    continue

            surf_area_region = (4 * np.pi * radii[i] ** 2) * (c_max - c_min) * (l_max - l_min) / (2 * np.pi * np.pi)
            n_samples = int(surf_area_region / (resolution[i] ** 2)) + 1
            x, y, z = self.adaptable_fibonacci_sphere(n_samples, c_min, c_max, l_min, l_max)

            self._x = np.append(self._x, x * radii[i])
            self._y = np.append(self._y, y * radii[i])
            self._z = np.append(self._z, z * radii[i])

    def discard_region(self, c_min, c_max, l_min, l_max, r_min, r_max):
        """
        This function removes a region from the grid
        :param c_min: minimum colatitude in radians
        :param c_max: maximum colatitude in radians
        :param l_min: minimum longitude in radians
        :param l_max: maximum longitude in radians
        :param r_min: minimum radius in km
        :param r_max: minimum radius in km
        :return:
        """

        r, c, l = cart2sph(self._x, self._y, self._z)
        pts = np.array((r, c, l)).T
        # Extract r_domain
        above_domain = pts[pts[:, 0] >= r_max]
        below_domain = pts[pts[:, 0] <= r_min]
        r_domain = pts[pts[:, 0] < r_max]
        r_domain = r_domain[r_domain[:, 0] > r_min]
        # Append points that fall outside of domain
        pts = np.append(above_domain, below_domain, axis=0)

        # Extract c_domain
        below_cmin = r_domain[r_domain[:, 1] <= c_min]
        above_cmax = r_domain[r_domain[:, 1] >= c_max]
        c_domain = r_domain[r_domain[:, 1] > c_min]
        c_domain = c_domain[c_domain[:, 1] < c_max]
        # Append points that fall outside of domain
        pts = np.append(pts, below_cmin, axis=0)
        pts = np.append(pts, above_cmax, axis=0)

        # Extract l_domain
        below_lmin = c_domain[c_domain[:, 2] <= l_min]
        above_lmax = c_domain[c_domain[:, 2] >= l_max]
        # Append points that fall outside of domain
        pts = np.append(pts, below_lmin, axis=0)
        pts = np.append(pts, above_lmax, axis=0)

        self._x, self._y, self._z = sph2cart(pts[:, 1], pts[:, 2], pts[:, 0])

    def adaptable_fibonacci_sphere(self, n_samples, c_min=0, c_max=np.pi, l_min=0, l_max=2*np.pi):
        """
        this function computes a regional fibonacci sphere
        :param n_samples:
        :param c_min:
        :param c_max:
        :param l_min:
        :param l_max:
        :return:
        """
        z_start = math.cos(c_min)
        z_end = math.cos(c_max)
        z_len = z_end - z_start
        part_of_circle = (l_max - l_min) / (2 * math.pi)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0)) * part_of_circle
        offset = z_len / n_samples
        def populate(idx):
            z = ((idx * offset) + z_start) + (offset / 2.0)
            r = math.sqrt(1 - z ** 2)
            phi = (idx * golden_angle) % (part_of_circle * 2 * math.pi) + l_min
            x = math.cos(phi) * r
            y = math.sin(phi) * r
            return x, y, z

        g = np.vectorize(populate)
        return np.fromfunction(g, (n_samples,))

    def get_coordinates(self, type='cartesian', is_normalised=False):
        """ This function returns the coordinates of the Fibonacci grid
        :param type: 'cartesian' or 'spherical'
        :param is_normalised: True or False
        :return: x, y, z or r, c, l
        """

        if type == 'spherical':
            r, c, l = cart2sph(self._x, self._y, self._z)
            if is_normalised:
                return r / self.r_earth, c, l
            else:
                return r, c, l
        elif type == 'cartesian':
            if is_normalised:
                return self._x / self.r_earth, self._y / self.r_earth, self._z / self.r_earth
            else:
                return self._x, self._y, self._z
        else:
            raise ValueError('Incorrect type specified in FibonacciGrid.get_coordinates')