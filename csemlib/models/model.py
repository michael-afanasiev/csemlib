import abc

import numpy as np
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from . import enclosing_elements
from meshpy.tet import build, Options, MeshInfo
from numba import jit
from scipy.spatial.ckdtree import cKDTree




class Model:
    """
    An abstract base class handling an external Earth model.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractproperty
    def data(self):
        pass

    @abc.abstractmethod
    def read(self):
        pass

    @abc.abstractmethod
    def write(self):
        pass


@jit(nopython=True)
def interpolate(idx, bry, dat):
    """Interpolate data over a list of elements.

    Given an array specifying the vertices of tetrahedra, and an array of interpolating barycentric
    coordinates, this function gets the correct indices in a parameter array defined on the vertices,
    and performs the interpolation.
    :param idx: An (i,j) array of indices. i->tetrahedral connectivity, j->number of interpolation points.
    :param bry: An (i,j) array of barycentric coordinates, mapped to the indices defined in idx.
    :param dat: A 1-D array of data, specified on all vertices. These values will be interpolated.
    :returns: An array of length j, containing interpolated values.
    """

    nx, ny = idx.shape
    result = np.zeros(ny)
    for i in range(ny):
        for j in range(nx):
            result[i] += dat[idx[j, i]] * bry[j, i]

    return result


def triangulate(x, y, z, true_x=None, true_y=None, true_z=None):
    """Triangulate a point cloud.

    Given a 3-D point cloud, defined by x, y, z, perform a Delauny triangulation
    and return a list of elements.
    :param x: (Cartesian) x value.
    :param y: (Cartesian) y value.
    :param z: (Cartesian) z value.
    :returns: A list-of-lists containing connectivity of the resulting mesh.
    """

    print("COMPUTING DELAUNY TRIANGULATION")
    # Set up the simplex vertices.
    pts = np.array((x, y, z)).T

    # Do the triangulation with MeshPy. Currently, this seems like the fastest way.
    mesh_info = MeshInfo()
    mesh_info.set_points(pts)
    opts = Options("Q")
    mesh = build(mesh_info, options=opts)
    # if len(true_x) and len(true_y) and len(true_z):
    #     mesh.set_points(np.array((true_x, true_y, true_z)).T)
    #     mesh.write_vtk("/Users/michaelafanasiev/Desktop/test.vtk", pts, tetra=mesh.elements)
    return mesh.elements


def write_vtk(filename, points, vals, tetra, name=None):
    """Writes a vtk from the given set of grid locations, values and connectivity

    :param points: An ndarray containing all the grid locations in cartesion coordinates
           in the form of:  | x0 y0 z0 |
                            | :  :  :  |
                            | xn yn zn |

    :param vals: an array containing the values to be specified on the mesh
    :param tetra: connectivity of the mesh

    """
    import pyvtk
    from pyvtk import PointData, Scalars
    vtkElements = pyvtk.VtkData(
                    pyvtk.UnstructuredGrid(
                    points,
                    tetra=tetra),
                    PointData(Scalars(vals, name)),
                    "Mesh")

    vtkElements.tofile(filename)


def shade(x_target, y_target, z_target, x_mesh, y_mesh, z_mesh, elements):
    """Perform Gourard shading over triangular elements.

    Given a 3-D point cloud, and a set of elements, compute the coefficients
    required to linearly interpolate parameter values from element vertices to
    their interior. This requires finding enclosing elements, and getting the barycentric
    coordinates of target points.
    :param x_target: (Cartesian) target x value for interpolation.
    :param y_target: (Cartesian) target y value for interpolation.
    :param z_target: (Cartesian) target z value for interpolation.
    :param x_mesh: (Cartesian) x values defining the tetrahedra.
    :param y_mesh: (Cartesian) y values defining the tetrahedra.
    :param z_mesh: (Cartesian) z values defining the tetrahedra.
    :param elements: List of lists defining the connectivity of tetrahedra.
    """

    print("COMPUTING GOURARD SHADING")
    # Generate KDTree of element vertices.
    tree = cKDTree(np.array((x_mesh, y_mesh, z_mesh)).T, balanced_tree=False)

    # Set the initial points to be found.
    query_points = np.array((x_target, y_target, z_target)).T

    # Initialize search parameters.
    radius, all_found, interp_values = 1, False, np.empty(query_points.shape[0])
    while not all_found:

        # Get closest 'radius' points
        _, f_points = tree.query(query_points, k=radius, n_jobs=-1)

        # Get homogeneous representation of found points.
        h_points = np.c_[query_points, np.ones(query_points.shape[0])]

        # Initialize array to hold points not yet found.
        not_found = np.array((), dtype=int)

        # Initialize dataframe.
        elements = np.array(elements)

        ind, bary = enclosing_elements.enclosing_elements(f_points.astype(np.int),
                                                          elements.astype(np.int),
                                                          h_points.astype(np.float),
                                                          x_mesh.astype(np.float),
                                                          y_mesh.astype(np.float),
                                                          z_mesh.astype(np.float))
        break

    return ind, bary
