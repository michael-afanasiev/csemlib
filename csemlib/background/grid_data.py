import pandas as pd
import numpy as np
import copy

from csemlib.background.fibonacci_grid import FibonacciGrid
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

    def set_coordinates(self, x, y, z):
        self.df['x'] = x
        self.df['y'] = y
        self.df['z'] = z

    def add_components(self, components):
        self.components.extend(components)
        for component in components:
            self.df[component] = np.zeros(len(self.df))

    def del_components(self, components):
        for component in components:
            del self.df[component]

    def set_component(self, component, values):
        if component not in self.df.columns:
            self.components.append(component)
        self.df[component] = values

    def get_component(self, component):
        return self.df[component].values

    def get_data(self):
        return self.df[self.components].values

    def get_coordinates(self, coordinate_type=None):
        if coordinate_type == 'spherical' and self.coordinate_system == 'cartesian':
            return cart2sph(*self.df[self.coordinates].values.T)
        elif coordinate_type == 'cartesian' and self.coordinate_system == 'spherical':
            return sph2cart(*self.df[self.coordinates].values.T)
        else:
            return self.df[self.coordinates].values

    def add_col_lon_rad(self):
        self.df['c'], self.df['l'], self.df['r'] = cart2sph(self.df['x'], self.df['y'], self.df['z'])

    def add_xyz(self):
        self.df['x'], self.df['y'], self.df['z'] = sph2cart(self.df['c'], self.df['l'], self.df['r'])

    def rotate(self, angle, x, y, z):
        rot_mat = get_rot_matrix(angle, x, y, z)
        self.df['x'], self.df['y'], self.df['z'] = rotate(self.df['x'], self.df['y'], self.df['z'], rot_mat)

        # Also update c,l,r coordinates
        self.add_col_lon_rad()