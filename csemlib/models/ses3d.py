import datetime
import io
import os

import numpy as np
import xarray
from meshpy.tet import MeshInfo, build, Options
from scipy.spatial import cKDTree

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

        # Set up the simplex vertices.
        pts = np.array((cols, lons, rads)).T

        # Set up the KdTree.
        tree = cKDTree(pts)

        # Do the triangulation with MeshPy.
        # I tried using packages included with SciPy, but they all seemed
        # too slow.
        mesh_info = MeshInfo()
        mesh_info.set_points(pts)
        opts = Options("Q")
        mesh = build(mesh_info, options=opts)

        # Set the initial points to be found.
        query_points = np.array((x, y, z)).T

        # Assume we won't find everything on the first pass
        radius = 1
        all_found = False
        interp_values = np.empty(query_points.shape[0])
        while not all_found:

            # Get closest 'radius' points
            _, f_points = tree.query(query_points, k=radius)

            # Get homogeneous representation of found points.
            h_points = np.c_[query_points, np.ones(query_points.shape[0])]

            # Initialize array to hold unfound points.
            not_found = np.array(())

            # Initialize dataframe.
            elem_array = np.array(mesh.elements)

            # Generate the vertex -> element mapping.
            vtx_to_element = [[] for _ in range(elem_array.shape[0])]
            for i in elem_array:
                for j in i:
                    vtx_to_element[j].append(i)

            # Pre make some index arrays.
            h = np.ones(4)
            idx = np.arange(4)
            par = self._data[param].values.ravel()

            # Loop through all elements found on this pass.
            for i, [target, vtx] in enumerate(zip(h_points, f_points)):

                # Assume we're not going to find the point
                found = False

                # Find all elements to which this vertex belongs.
                candidates = vtx_to_element[vtx]

                for elem in candidates:

                    # Setup (homogeneous) representation of candidate vertex.
                    x = cols[elem[idx]]
                    y = lons[elem[idx]]
                    z = rads[elem[idx]]
                    t = np.array([x, y, z, h])

                    # Find barycentric coordinates and test for containment.
                    bary = np.linalg.solve(t, target.T)
                    if np.all(np.logical_and(bary >= 0, bary <= 1)):
                        vtx_vals = par[elem[idx]]

                        # Interpolate value if found.
                        interp_values[i] = np.dot(bary, vtx_vals)

                        # If we find the point, no more work needs to be done
                        found = True
                        break

                # Save those points which we do not find.
                if not found:
                    not_found = np.append(not_found, [i])

            # Recreate missed query points or quit.
            if not_found.size:
                query_points = np.atleast_2d(
                    np.array((x[not_found], y[not_found], z[not_found])).T)
            else:
                all_found = True

        return interp_values
