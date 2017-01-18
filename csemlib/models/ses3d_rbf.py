import io
import os

import sys

from csemlib.background.grid_data import GridData
from csemlib.models.ses3d import Ses3d

import numpy as np
import scipy.spatial as spatial
from scipy.interpolate import Rbf
from scipy.spatial.qhull import ConvexHull
from scipy.interpolate import griddata



class Ses3d_rbf(Ses3d):
    """
    Class built open Ses3D which adds extra interpolation methods
    """

    def __init__(self, name, directory, components=[], doi=None, interp_method='nearest_neighbour'):
        super(Ses3d_rbf, self).__init__(name, directory, components, doi)
        self.read()
        self.grid_data_ses3d = GridData()
        self.init_grid_data()
        self.interp_method = interp_method

    def split_domain(self, GridData):
        ses3d_pts = self.grid_data_ses3d.get_coordinates(coordinate_type='cartesian')

        # Collect all the points that form the convex hull
        hull = ConvexHull(ses3d_pts)
        pts_hull = []
        for point in np.unique(hull.simplices.flatten()):
            pts_hull.append(ses3d_pts[point])
        pts_hull = np.array(pts_hull)

        # Perform Delauney triangulation from hull points
        hull = spatial.Delaunay(pts_hull)

        # Split points into points that fall inside and outside of convex hull
        in_or_out = hull.find_simplex(GridData.get_coordinates(coordinate_type='cartesian'))>=0
        indices_in = np.where(in_or_out == True)
        indices_out = np.where(in_or_out == False)

        pts_other = GridData[indices_out]
        pts_new = GridData[indices_in]

        return pts_new, pts_other


    def init_grid_data(self):
        x = self.data()['x'].values.ravel()
        y = self.data()['y'].values.ravel()
        z = self.data()['z'].values.ravel()
        self.grid_data_ses3d = GridData(x, y, z, components=self.components)

        for component in self.components:
            self.grid_data_ses3d.set_component(component, self.data()[component].values.ravel())


    def eval_point_cloud_griddata(self, GridData):
        print('Evaluating SES3D model:', self.model_info['model'])

        grid_coords = self.grid_data_ses3d.get_coordinates(coordinate_type='cartesian')
        # Split domain in points that lie within convex hull and fall outside
        ses3d_dmn = self.extract_ses3d_dmn(GridData)

        # Generate KDTrees
        pnt_tree_orig = spatial.cKDTree(grid_coords)

        # do nearest neighbour
        if self.interp_method == 'nearest_neighbour':
            _, indices = pnt_tree_orig.query(ses3d_dmn.get_coordinates(coordinate_type='cartesian'), k=1)
            for component in self.components:
                if self.model_info['component_type'] == 'perturbation':
                    ses3d_dmn.df[:][component] += self.grid_data_ses3d.df[component][indices].values
                if self.model_info['component_type'] == 'absolute':
                    ses3d_dmn.df[:][component] += self.grid_data_ses3d.df[component][indices].values

            GridData.df.update(ses3d_dmn.df)
            return

        # Use 20 nearest points
        _, all_neighbours = pnt_tree_orig.query(ses3d_dmn.get_coordinates(coordinate_type='cartesian'), k=50)

        # Interpolate ses3d value for each grid point
        i = 0
        for neighbours in all_neighbours:
            x_c_orig, y_c_orig, z_c_orig = grid_coords[neighbours].T
            for component in self.components:
                dat_orig = self.grid_data_ses3d.df[component][neighbours].values
                coords_new = ses3d_dmn.get_coordinates(coordinate_type='cartesian').T
                x_c_new, y_c_new, z_c_new = coords_new.T[i]

                if self.interp_method == 'griddata_linear':
                    pts_local = np.array((x_c_orig, y_c_orig, z_c_orig)).T
                    xi = np.array((x_c_new, y_c_new, z_c_new))
                    val = griddata(pts_local, dat_orig, xi, method='linear', fill_value=0.0)
                elif self.interp_method == 'radial_basis_func':
                    rbfi = Rbf(x_c_orig, y_c_orig, z_c_orig, dat_orig, function='linear')
                    val = rbfi(x_c_new, y_c_new, z_c_new)

                if self.model_info['component_type'] == 'perturbation':
                    ses3d_dmn.df[component].values[i] += val
                elif self.model_info['component_type'] == 'absolute':
                    ses3d_dmn.df[component].values[i] = val
            i += 1

            if i % 200 == 0:
                ind = float(i)
                percent = ind/len(all_neighbours)*100.0
                sys.stdout.write("\rProgress: %.1f%%" % percent)
                sys.stdout.flush()

        GridData.df.update(ses3d_dmn.df)
        return GridData
