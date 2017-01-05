from csemlib.background.grid_data import GridData
from csemlib.models.ses3d import Ses3d

import numpy as np
import scipy.spatial as spatial
from scipy.interpolate import Rbf
from scipy.spatial.qhull import ConvexHull
from scipy.interpolate import griddata



class Ses3d_rbf(Ses3d):
    """
    Class handling file-IO for a model in SES3D format.
    """

    def __init__(self, name, directory, components=[],
                 rotation_vector=None, rotation_angle=None, doi=None):
        super(Ses3d_rbf, self).__init__(name, directory, components,
                 rotation_vector, rotation_angle, doi)
        self.read()
        self.grid_data_ses3d = GridData()
        self.init_grid_data()
        self.interp_method = 'griddata_linear'
        self.model_type = 'perturbation_percent'

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
        x = self.data['x'].values.ravel()
        y = self.data['y'].values.ravel()
        z = self.data['z'].values.ravel()
        self.grid_data_ses3d = GridData(x, y, z, components=self.components)

        for component in self.components:
            self.grid_data_ses3d.set_component(component, self.data[component].values.ravel())

    def eval_point_cloud(self, GridData):
        grid_coords = self.grid_data_ses3d.get_coordinates(coordinate_type='cartesian')

        # Split domain in points that lie within convex hull and fall outside
        grid_inside, grid_outside = self.split_domain(GridData)

        # Generate KDTrees
        pnt_tree_orig = spatial.cKDTree(grid_coords)

        # Use 20 nearest points
        _, pairs = pnt_tree_orig.query(grid_inside.get_coordinates(coordinate_type='cartesian'), k=20)

        # Interpolate ses3d value for each grid point
        i = 0
        for idx in pairs:
            x_c_orig, y_c_orig, z_c_orig = grid_coords[idx].T
            for component in self.components:
                dat_orig = self.grid_data_ses3d.df[component][idx].values


                coords_new = grid_inside.get_coordinates(coordinate_type='cartesian').T
                x_c_new, y_c_new, z_c_new = coords_new.T[i]

                if self.interp_method == 'griddata_linear':
                    pts_local = np.array((x_c_orig, y_c_orig, z_c_orig)).T
                    xi = np.array((x_c_new, y_c_new, z_c_new))
                    val = griddata(pts_local, dat_orig, xi, method='linear', fill_value=0.0)
                else:
                    rbfi = Rbf(x_c_orig, y_c_orig, z_c_orig, dat_orig, function='linear')
                    val = rbfi(x_c_new, y_c_new, z_c_new)

                if self.model_type == 'perturbation_percent':
                    grid_inside.df[component[1:]][i] *= (1 + val/100.0)
                elif self.model_type == 'absolute':
                    grid_inside.df[component[1:]][i] = val
            i += 1

            if i % 200 == 0:
                print(i)

        grid_inside.append(grid_outside)
        return grid_inside