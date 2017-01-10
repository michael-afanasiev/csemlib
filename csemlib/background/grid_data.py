import pandas as pd
import numpy as np
import copy

from csemlib.background.fibonacci_grid import FibonacciGrid
from csemlib.models.s20rts import S20rts
from csemlib.utils import cart2sph, sph2cart


class GridData:
    """
    Class that serves as a collection point of information on the grid,
    its coordinates and the corresponding data.
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
        if self.coordinates is not ['c', 'l', 'r']:
            self.df['c'], self.df['l'], self.df['r'] = cart2sph(self.df['x'], self.df['y'], self.df['z'])



# # Generate visualisation grid
# fib_grid = FibonacciGrid()
# # Set global background grid
# radii = np.linspace(6250.0, 0.0, 15)
# resolution = np.ones_like(radii) * (6350.0 / 15)
# fib_grid.set_global_sphere(radii, resolution)
# # refinement region coarse
# c_min = np.radians(30)
# c_max = np.radians(70)
# l_min = np.radians(120)
# l_max = np.radians(160)
# radii_regional = np.linspace(6250.0, 6150.0, 4)
# resolution_regional = np.ones_like(radii_regional) * 50
# fib_grid.add_refinement_region(c_min, c_max, l_min, l_max, radii_regional, resolution_regional)
#
#
# s20 = S20rts()
#
# # Setup GridData
# grid_data = GridData(*fib_grid.get_coordinates())
#
# grid_data_copy = grid_data.copy()
#
#
# s20dmn = s20.eval_point_cloud_griddata(grid_data)
#
# grid_data.add_col_lon_rad()
#
#
# print(grid_data.df)