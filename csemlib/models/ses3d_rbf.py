import os

import sys

import h5py
from csemlib.background.grid_data import GridData
from csemlib.models.ses3d import Ses3d

import numpy as np
import scipy.spatial as spatial
from scipy.interpolate import Rbf
from scipy.interpolate import griddata


class Ses3d_rbf(Ses3d):
    """
    Class built open Ses3D which adds extra interpolation methods
    """

    def __init__(self, name, directory, components=[], doi=None, interp_method='nearest_neighbour'):
        super(Ses3d_rbf, self).__init__(name, directory, components, doi)
        self.grid_data_ses3d = None
        self.interp_method = interp_method

    def init_grid_data_hdf5(self, region=0):
        filename = os.path.join(self.directory, "{}.hdf5".format(self.model_info['model']))
        f = h5py.File(filename, "r")

        x = f['region_{}'.format(region)]['x'][:]
        y = f['region_{}'.format(region)]['y'][:]
        z = f['region_{}'.format(region)]['z'][:]

        self.grid_data_ses3d = GridData(x, y, z, components=self.components)

        if self.model_info['taper']:
            components = self.components + ['taper']
        else:
            components = self.components
        for component in components:
            self.grid_data_ses3d.set_component(component, f['region_{}'.format(region)][component][:])
        f.close()

    def eval_point_cloud_griddata(self, GridData, interp_method=None):
        print('Evaluating SES3D model:', self.model_info['model'])
        interp_method = interp_method or self.interp_method

        # make a loop here to go through each region one at a time
        for region in range(self.model_info['region_info']['num_regions']):
            # Get points that lie within region
            ses3d_dmn = self.extract_ses3d_dmn(GridData, region)
            # if no points in region, continue
            if len(ses3d_dmn) == 0:
                continue

            self.init_grid_data_hdf5(region)
            grid_coords = self.grid_data_ses3d.get_coordinates(coordinate_type='cartesian')

            # Generate KDTrees
            pnt_tree_orig = spatial.cKDTree(grid_coords, balanced_tree=False)

            # Do nearest neighbour
            if interp_method == 'nearest_neighbour':
                self.nearest_neighbour_interpolation(pnt_tree_orig, ses3d_dmn, GridData)
            else:
                self.grid_and_rbf_interpolation(pnt_tree_orig, ses3d_dmn, interp_method, grid_coords, GridData)


    def nearest_neighbour_interpolation(self, pnt_tree_orig, ses3d_dmn, GridData):
        _, indices = pnt_tree_orig.query(ses3d_dmn.get_coordinates(coordinate_type='cartesian'), k=1)
        for component in self.components:
            if self.model_info['component_type'] == 'perturbation':
                if self.model_info['taper']:
                    taper = self.grid_data_ses3d.df['taper'][indices].values
                    one_d = ses3d_dmn.df[:]['one_d_{}'.format(component)]
                    ses3d_dmn.df[:][component] = ((one_d + self.grid_data_ses3d.df[component][indices].values) * taper) + \
                                                 (1 - taper) * ses3d_dmn.df[:][component]
                else:
                    ses3d_dmn.df[:][component] += self.grid_data_ses3d.df[component][indices].values
            if self.model_info['component_type'] == 'absolute':
                if self.model_info['taper']:
                    taper = self.grid_data_ses3d.df['taper'][indices].values
                    ses3d_dmn.df[:][component] = taper * self.grid_data_ses3d.df[component][indices].values +\
                                                 (1 - taper) * ses3d_dmn.df[:][component]
                else:
                    ses3d_dmn.df[:][component] = self.grid_data_ses3d.df[component][indices].values

        GridData.df.update(ses3d_dmn.df)


    def grid_and_rbf_interpolation(self, pnt_tree_orig, ses3d_dmn, interp_method, grid_coords, GridData):
        # Use 30 nearest points
        _, all_neighbours = pnt_tree_orig.query(ses3d_dmn.get_coordinates(coordinate_type='cartesian'), k=100)

        # Interpolate ses3d value for each grid point
        i = 0
        if self.model_info['taper']:
            components = ['taper'] + self.components
            ses3d_dmn.set_component('taper', np.zeros(len(ses3d_dmn)))
        else:
            components = self.components

        for neighbours in all_neighbours:
            x_c_orig, y_c_orig, z_c_orig = grid_coords[neighbours].T
            for component in components:
                dat_orig = self.grid_data_ses3d.df[component][neighbours].values
                coords_new = ses3d_dmn.get_coordinates(coordinate_type='cartesian').T
                x_c_new, y_c_new, z_c_new = coords_new.T[i]

                if interp_method == 'griddata_linear':
                    pts_local = np.array((x_c_orig, y_c_orig, z_c_orig)).T
                    xi = np.array((x_c_new, y_c_new, z_c_new))
                    if self.model_info['component_type'] == 'absolute':
                        val = griddata(pts_local, dat_orig, xi, method='linear',
                                       fill_value=ses3d_dmn.df[component].values[i])
                    elif self.model_info['component_type'] == 'perturbation':
                        val = griddata(pts_local, dat_orig, xi, method='linear', fill_value=0.0)

                elif interp_method == 'radial_basis_func':
                    rbfi = Rbf(x_c_orig, y_c_orig, z_c_orig, dat_orig, function='linear')
                    val = rbfi(x_c_new, y_c_new, z_c_new)

                if self.model_info['component_type'] == 'perturbation':
                    if self.model_info['taper'] and component != 'taper':
                        taper = ses3d_dmn.df['taper'].values[i]
                        one_d = ses3d_dmn.df['one_d_{}'.format(component)].values[i]
                        ses3d_dmn.df[component].values[i] += (taper * val)
                        ses3d_dmn.df[component].values[i] = (one_d + val) * taper + \
                                                            (1 - taper) * ses3d_dmn.df[component].values[i]
                    else:
                        ses3d_dmn.df[component].values[i] += val
                elif self.model_info['component_type'] == 'absolute':
                    if self.model_info['taper'] and component != 'taper':
                        taper = ses3d_dmn.df['taper'].values[i]
                        ses3d_dmn.df[component].values[i] = taper * val + (1-taper) * ses3d_dmn.df[component].values[i]
                    else:
                        ses3d_dmn.df[component].values[i] = val
            i += 1

            if i % 200 == 0:
                ind = float(i)
                percent = ind / len(all_neighbours) * 100.0
                sys.stdout.write("\rProgress: %.1f%%" % percent)
                sys.stdout.flush()

        if self.model_info['taper']:
            del ses3d_dmn.df['taper']

        GridData.df.update(ses3d_dmn.df)
