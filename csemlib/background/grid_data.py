import pandas as pd
import numpy as np
import copy

from csemlib.models.one_dimensional import prem_eval_point_cloud
from csemlib.utils import cart2sph, sph2cart, get_rot_matrix, rotate


class GridData:
    """
    Class that serves as a collection point of information on the grid,
    its coordinates and the corresponding data. Data and coordinates must have the same length.
    """

    def __init__(self, x=[], y=[], z=[], components=[], coord_system='cartesian'):
        self.coordinate_system = coord_system
        self.components = []
        if self.coordinate_system == 'cartesian':
            self.coordinates = ['x', 'y', 'z']
        elif self.coordinate_system == 'spherical':
            self.coordinates = ['c', 'l', 'r']
        self.df = pd.DataFrame(np.array((x, y, z)).T, columns=self.coordinates)
        self.add_components(components)

        if self.coordinate_system == 'cartesian':
            self.add_col_lon_rad()
        elif self.coordinate_system == 'spherical':
            self.add_xyz()

    def __getitem__(self, i):
        x, y, z = self.df[self.coordinates].loc[i].values.T
        grid_data = GridData(x, y, z, coord_system=self.coordinate_system)
        for component in self.components:
            grid_data.set_component(component, self.df[component].loc[i].values)
        return grid_data

    def __len__(self):
        return len(self.df)

    def copy(self):
        return copy.deepcopy(self)

    def append(self, griddata):
        self.df = self.df.append(griddata.df)

    def add_components(self, components):
        self.components.extend(components)
        for component in components:
            self.df[component] = np.zeros(len(self.df))

    def del_components(self, components):
        for component in components:
            del self.df[component]
            self.components.remove(component)

    def set_component(self, component, values):
        if component not in self.df.columns:
            self.components.append(component)
        self.df[component] = values

    def get_component(self, component):
        return self.df[component].values

    def get_data(self):
        return self.df[self.components].values

    def get_coordinates(self, coordinate_type=None):
        coordinate_type = coordinate_type or self.coordinate_system

        if coordinate_type == 'spherical':
            return self.df[['c', 'l', 'r']].values
        elif coordinate_type == 'cartesian':
            return self.df[['x', 'y', 'z']].values

    def add_col_lon_rad(self):
        self.df['c'], self.df['l'], self.df['r'] = cart2sph(self.df['x'], self.df['y'], self.df['z'])

    def add_xyz(self):
        self.df['x'], self.df['y'], self.df['z'] = sph2cart(self.df['c'], self.df['l'], self.df['r'])

    def rotate(self, angle, x, y, z):
        rot_mat = get_rot_matrix(angle, x, y, z)
        self.df['x'], self.df['y'], self.df['z'] = rotate(self.df['x'], self.df['y'], self.df['z'], rot_mat)

        # Also update c,l,r coordinates
        self.add_col_lon_rad()

    def add_one_d(self):
        one_d_rho, one_d_vpv, one_d_vsv, one_d_vsh = prem_eval_point_cloud(self.df['r'])
        self.df['one_d_rho'] = one_d_rho
        self.df['one_d_vp'] = one_d_vpv
        self.df['one_d_vsv'] = one_d_vsv
        self.df['one_d_vsh'] = one_d_vsh

    def del_one_d(self):
        one_d_parameters = ['one_d_rho', 'one_d_vp', 'one_d_vsv', 'one_d_vsh']
        for param in one_d_parameters:
            del self.df[param]
