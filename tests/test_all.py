import os

import numpy as np
import pytest
import xarray
from meshpy.tet import MeshInfo, build, Options

import csemlib
import csemlib.background.skeleton as skl
import csemlib.models.crust as crust
import csemlib.models.one_dimensional as m1d
import csemlib.models.s20rts as s20
import csemlib.models.ses3d as s3d
from csemlib.models.model import triangulate, write_vtk
from csemlib.models.topography import Topography
from csemlib.utils import cart2sph, sph2cart

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')
DECIMAL_CLOSE = 3


def test_fibonacci_sphere():
    true_y = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    true_x = np.array([0.43588989, -0.52658671, 0.0757129, 0.58041368, -0.97977755,
                       0.83952592, -0.24764672, -0.39915719, 0.67080958, -0.40291289])
    true_z = np.array([0., 0.48239656, -0.86270943, 0.75704687, -0.17330885,
                       -0.53403767, 0.92123347, -0.76855288, 0.24497858, 0.16631658])

    points = skl.fibonacci_sphere(10)
    np.testing.assert_almost_equal(points[0], true_x, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(points[1], true_y, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(points[2], true_z, decimal=DECIMAL_CLOSE)


def test_prem_no220():
    """
    Test to (mainly) make sure that discontinuities are handled properly.
    :return:
    """
    reg01 = np.array([3.374814283471982, 8.076866567257888, 4.4708837074242656, 4.570983707424266])
    reg02 = np.array([3.363837607910846, 8.014433950714174, 4.433659080207189, 4.422659080207189])
    reg03 = np.array([3.495466943964841, 8.751316606498193, 4.7139374352534915, 4.7139374352534915])
    reg04 = np.array([3.5432636006906293, 8.905243242819024, 4.7699, 4.7699])
    reg05 = np.array([3.7237469157118186, 9.133916669282687, 4.932487458797674, 4.932487458797674])
    reg06 = np.array([3.9758203735677293, 10.157825003924035, 5.5159311881965145, 5.5159311881965145])
    reg07 = np.array([3.9921213467273584, 10.266174462407786, 5.570211034374509, 5.570211034374509])
    reg08 = np.array([4.3807429838542795, 10.751407424647873, 5.945126361510368, 5.945126361510368])
    reg09 = np.array([4.443204194391242, 11.06568986974271, 6.240535840301453, 6.240535840301453])
    reg10 = np.array([5.491476554415982, 13.680424477483925, 7.2659252231153015, 7.2659252231153015])
    reg11 = np.array([5.566455445926154, 13.716622269026377, 7.26465059504689, 7.26465059504689])
    reg12 = np.array([9.903438401183957, 8.064788053141768, 0.0, 0.0])
    reg13 = np.array([12.166331854652926, 10.35571579802768, 0.0, 0.0])
    reg14 = np.array([12.763614264456663, 11.02826139091006, 3.5043113193074316, 3.5043113193074316])

    np.testing.assert_almost_equal(m1d.prem_no220(6292, region='upper_mantle'), reg01, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(6191, region='upper_mantle'), reg02, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(6051, region='upper_mantle'), reg03, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(5971, region='upper_mantle'), reg04, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(5971, region='transition_zone'), reg05, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(5771, region='transition_zone'), reg06, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(5701, region='transition_zone'), reg07, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(5701, region='lower_mantle'), reg08, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(5600, region='lower_mantle'), reg09, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(3630, region='lower_mantle'), reg10, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(3480, region='lower_mantle'), reg11, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(3480, region='outer_core'), reg12, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(1221.5, region='outer_core'), reg13, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(m1d.prem_no220(1221.5, region='inner_core'), reg14, decimal=DECIMAL_CLOSE)

    # Make sure that questionable requests will error out.
    with pytest.raises(ValueError):
        m1d.prem_no220(6371, 'uppermantle')
    with pytest.raises(ValueError):
        m1d.prem_no220(5971, 'lower_mantle')


def test_crust():
    """
    Test to ensure that the crust returns correct values.
    """

    proper_dep = np.array([[38.69471863, 17.96798953],
                           [38.69471863, 17.96798953]])
    proper_vs = np.array([[3.64649739, 3.1255109],
                          [3.64649739, 3.1255109]])

    cst = crust.Crust()
    cst.read()

    x = np.radians([179, 1])
    y = np.radians([1, 1])
    lats, lons = np.meshgrid(x, y)
    vals_dep = cst.eval(lats, lons, param='crust_dep')
    vals_vs = cst.eval(lats, lons, param='crust_vs')

    np.testing.assert_almost_equal(vals_dep, proper_dep, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(vals_vs, proper_vs, decimal=DECIMAL_CLOSE)


def test_gourard_shading():
    """
    Test to see if the interpolation function works over a tetrahedra.
    """

    true_val = 4
    data = np.array([2, 2, 2, 4]).T
    bry = np.array([[0.5, 0.5, 0.5, 0.25]]).T
    idx = np.array([[0, 1, 2, 3]]).T

    np.testing.assert_almost_equal(
        csemlib.models.model.interpolate(idx, bry, data), true_val, decimal=DECIMAL_CLOSE)


def test_barycenter_detection():
    """
    Simple test to ensure that the interpolation routine works.
    """

    true_ind = np.array([[0, 0], [3, 3], [7, 7], [2, 2]], dtype=np.int64)
    true_bar = np.array([[1.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00],
                         [0.00000000e+00, 1.00000000e+00],
                         [0.00000000e+00, 0.00000000e+00]])

    vertices = [
        (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
        (0, 0, 12), (2, 0, 12), (2, 2, 12), (0, 2, 12),
    ]
    x_mesh, y_mesh, z_mesh = np.array(vertices)[:, 0], np.array(vertices)[:, 1], np.array(vertices)[:, 2]
    x_target, y_target, z_target = np.array([0, 0]), np.array([0, 2]), np.array([0, 12])
    mesh_info = MeshInfo()
    mesh_info.set_points(list(vertices))
    mesh_info.set_facets([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ])
    opts = Options("Q")
    mesh = build(mesh_info, options=opts)
    elements = np.array(mesh.elements)
    ind, bary = csemlib.models.model.shade(x_target, y_target, z_target,
                                           x_mesh, y_mesh, z_mesh,
                                           elements)
    np.testing.assert_almost_equal(ind, true_ind, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(bary, true_bar, decimal=DECIMAL_CLOSE)


def test_ses3d():
    """
    Test to ensure that a ses3d model returns itself.
    """

    mod = s3d.Ses3d('japan', os.path.join(TEST_DATA_DIR, 'japan'),
                    components=['drho', 'dvsv', 'dvsh', 'dvp'])
    mod.read()

    all_cols, all_lons, all_rads = np.meshgrid(
        mod.data.coords['col'].values,
        mod.data.coords['lon'].values,
        mod.data.coords['rad'].values)
    interp = mod.eval(mod.data['x'].values.ravel(), mod.data['y'].values.ravel(),
                      mod.data['z'].values.ravel(), param=['dvsv', 'drho', 'dvsh', 'dvp'])
    # Setup true data.
    true = np.empty((len(all_cols.ravel()), 4))
    true[:, 0] = mod.data['dvsv'].values.ravel()
    true[:, 1] = mod.data['drho'].values.ravel()
    true[:, 2] = mod.data['dvsh'].values.ravel()
    true[:, 3] = mod.data['dvp'].values.ravel()

    np.testing.assert_almost_equal(true, interp, decimal=DECIMAL_CLOSE)


def test_s20rts():
    """
    Test to ensure that s20 rts calls returns some proper values.
    :return:
    """

    true = np.array([
        [-1.01877232e-02, -1.01877232e-02, -1.01877232e-02, -1.01877232e-02, -1.01877232e-02,
         -1.01877232e-02, -1.01877232e-02, -1.01877232e-02, -1.01877232e-02, -1.01877232e-02],
        [-1.44056273e-02, 2.96664697e-02, 2.92642415e-02, 1.61460041e-02, -1.57275509e-02,
         7.13132098e-03, 2.90914878e-02, 2.99952254e-02, 5.79711610e-03, -1.44056273e-02],
        [3.37437506e-02, 2.04684792e-02, -1.24400607e-02, 2.72054351e-03, 8.33735766e-03,
         1.18519683e-02, 5.28123669e-03, 2.79334604e-02, 1.14312565e-02, 3.37437506e-02],
        [1.88464920e-02, -2.84314823e-03, 9.61633282e-03, 3.41507489e-02, 2.75727421e-02,
         1.68134034e-02, -1.60801974e-02, 3.64814426e-02, -4.61877955e-03, 1.88464920e-02],
        [3.49627974e-02, -2.25299414e-02, 5.38457115e-03, -1.28656351e-02, 2.23747856e-02,
         1.37116499e-02, -1.02294856e-02, -9.12301242e-03, 4.90924855e-03, 3.49627974e-02],
        [1.52122538e-02, 1.95654151e-02, -1.82730716e-03, 1.83242680e-03, -3.33209386e-02,
         2.42266632e-02, -2.14003047e-02, 4.65260346e-03, 3.98520761e-02, 1.52122538e-02],
        [1.77482107e-03, 1.45018273e-02, -2.46039369e-02, 3.74249736e-02, -6.59335407e-03,
         1.66440321e-02, -2.50129693e-02, -1.12087136e-02, 2.13203960e-02, 1.77482107e-03],
        [-1.76785861e-02, -4.01646331e-04, -2.15678403e-02, -2.20824982e-02, -1.08647419e-02,
         -2.65258612e-03, -3.65854079e-02, -1.95070464e-03, 1.47419745e-02, -1.76785861e-02],
        [1.03761537e-02, 1.48621690e-02, 1.61364041e-02, 2.67424633e-02, -1.33043420e-02,
         -2.34031725e-02, 5.95206701e-04, -4.95703024e-03, -9.53130089e-05, 1.03761537e-02],
        [5.83822775e-03, 5.83822775e-03, 5.83822775e-03, 5.83822775e-03, 5.83822775e-03,
         5.83822775e-03, 5.83822775e-03, 5.83822775e-03, 5.83822775e-03, 5.83822775e-03]])

    mod = s20.S20rts()
    mod.read()

    size = 10
    col = np.linspace(0, np.pi, size)
    lon = np.linspace(0, 2 * np.pi, size)
    cols, lons = np.meshgrid(col, lon)
    rad = mod.layers[0]

    vals = mod.eval(cols, lons, rad, 'test').reshape(size, size).T
    dat = xarray.DataArray(vals, dims=['lat', 'lon'], coords=[90 - np.degrees(col), np.degrees(lon)])
    np.testing.assert_almost_equal(dat.values, true, decimal=DECIMAL_CLOSE)


def test_s20rts_vtk_single_sphere():
    """
    Test to ensure that a vtk of s20rts is written succesfully.
    :return:


    """
    s20mod = s20.S20rts()
    s20mod.read()

    rad = s20mod.layers[0]
    rel_rad = rad/ s20mod.r_earth
    x, y, z = skl.fibonacci_sphere(500)
    _, c, l = cart2sph(x, y, z)
    vals = s20mod.eval(c, l, rad, 'test')

    elements = triangulate(x,y,z)

    pts = np.array((x, y, z)).T * rel_rad
    write_vtk("test_s20rts.vtk", pts, elements, vals, 'vs')


def test_3d_s20rts_vtk():
    """
    Test to ensure that a vtk of s20rts is written succesfully.
    :return:

    """
    s20mod = s20.S20rts()
    s20mod.read()

    radii = [6346.63, 6296.63, 6241.64, 6181.14, 6114.57, 6041.34, 5960.79, 5872.18, 5774.69, 5667.44, 5549.46,
              5419.68,5276.89, 5119.82, 4947.02, 4756.93, 4547.81, 4317.74, 4064.66, 3786.25, 3479.96]

    n_samples = 1000
    x, y, z = skl.multiple_fibonacci_spheres(radii, n_samples)
    r, c, l = cart2sph(x, y, z)

    c = c[0:n_samples]
    l = l[0:n_samples]

    # all c and r are the same if the spheres are created with equal amounts of points
    vals = s20mod.eval(c,l,radii[0], 'test')

    for i in range(len(radii))[1:]:
        new_vals = s20mod.eval(c,l, radii[i], 'test')
        vals = np.append(vals, new_vals)

    elements = triangulate(x,y,z)
    pts = np.array((x, y, z)).T
    write_vtk("test_3d_s20rts.vtk", pts, elements, vals, 'vs')


def test_s20rts_point_cloud():
    """
    Test to ensure that a vtk of s20rts is written succesfully.
    :return:

    """
    # Initialize s20rts
    s20mod = s20.S20rts()
    s20mod.read()

    # Generate point cloud
    n_samples = 20
    n_layers = 10
    radii = np.linspace(s20mod.layers[0], s20mod.layers[-1], n_layers)
    x, y, z = skl.multiple_fibonacci_spheres(radii, n_samples)
    r, c, l = cart2sph(x, y, z)

    # Evaluate points
    c, l, r, vals = s20mod.eval_point_cloud(c, l, r, 'test')

    # Generate coordinates and mesh connectivity
    x, y, z = sph2cart(c, l, r)
    elements = triangulate(x,y,z)

    # Write to vtk
    pts = np.array((x, y, z)).T
    write_vtk("test_3d_s20rts.vtk", pts, elements, vals, 'vs')

def test_s20rts_out_of_bounds():
    mod = s20.S20rts()

    with pytest.raises(ValueError):
        mod.find_layer_idx(3200)

    with pytest.raises(ValueError):
        mod.find_layer_idx(6204)

    with pytest.raises(ValueError):
        mod.eval(0, 0, 7000, 'test')

def test_add_crust_to_prem():
    # Generate point cloud
    n_samples = 20
    n_layers = 10
    radii = np.linspace(6371.0, 0.0, n_layers)
    r_earth = 6371.0

    x, y, z = skl.multiple_fibonacci_spheres(radii, n_samples, normalized_radius=False)
    r, c, l = cart2sph(x, y, z)

    # Evaluate Prem
    rho, vpv, vsv, vsh = m1d.prem_eval_point_cloud(r)

    # Split into crustal and non crustal zone
    pts = np.array((c, l, r, rho, vpv, vsv, vsh)).T
    cst_zone = pts[pts[:, 2] >= (r_earth - 100.0)]
    non_cst_zone = pts[pts[:, 2] < (r_earth - 100.0)]

    # Compute crustal depths and vs for crustal zone coordinates
    cst = crust.Crust()
    cst.read()
    crust_dep = cst.eval(cst_zone[:, 0], cst_zone[:, 1], param='crust_dep', crust_smooth_factor=1)
    crust_vs = cst.eval(cst_zone[:, 0], cst_zone[:, 1], param='crust_vs', crust_smooth_factor=1)

    cst_zone[:, 5] = crust.add_crust(cst_zone[:, 2], crust_dep, crust_vs, cst_zone[:, 5])

    # Append crustal and non crustal zone back together
    pts = np.append(cst_zone, non_cst_zone, axis=0)

    # Generate mesh for plotting
    x, y, z = sph2cart(pts[:, 0], pts[:, 1], pts[:, 2]/6371.0)
    elements = triangulate(x, y, z)

    # Write to vtk
    points = np.array((x, y, z)).T
    write_vtk("crust_vs.vtk", points, elements, pts[:, 5], 'vs')

def test_vectorized_prem():
    # Generate point cloud
    n_samples = 5
    n_layers = 100
    radii = np.linspace(6371.0, 0.0, n_layers)


    x, y, z = skl.multiple_fibonacci_spheres(radii, n_samples, normalized_radius=False)
    r, c, l = cart2sph(x, y, z)

    rho, vpv, vsv, vsh = m1d.prem_eval_point_cloud(r)

    with pytest.raises(ValueError):
        m1d.prem_eval_point_cloud([-1.0])


def test_add_crust_and_s20rts_prem():
    """
    Test where both s20rts and the crust are added to prem, this test does not include
    topography yet.
    """

    # Generate point cloud based on average distance to the next point
    num_layers = 10
    radii = np.linspace(6371.0, 0.0, num_layers)
    r_earth = 6371.0
    res = r_earth / num_layers

    x, y, z = skl.multiple_fibonacci_resolution(radii, resolution=res, min_samples=10)
    r, c, l = cart2sph(x, y, z)

    # Evaluate Prem
    rho, vpv, vsv, vsh = m1d.prem_eval_point_cloud(r)
    pts = np.array((c, l, r, rho, vpv, vsv, vsh))

    # Evaluate s20rts
    s20mod = s20.S20rts()
    pts = s20mod.eval_point_cloud_non_norm(*pts)

    cst = crust.Crust()
    pts = cst.eval_point_cloud_with_topo_all_params(*pts.T)

    # Generate mesh for plotting (normalised coordinates)
    x, y, z = sph2cart(pts[:, 0], pts[:, 1], pts[:, 2]/ r_earth)
    elements = triangulate(x, y, z)
    coords = np.array((x, y, z)).T

    # Write to vtk
    write_vtk("crust_vsv.vtk", coords, elements, pts[:, 5], 'vsv')


def test_topo():
    """
    Test to ensure that a vtk of the topography is written succesfully.
    :return:

    """
    topo = Topography()
    topo.read()

    x, y, z = skl.fibonacci_sphere(10000)
    _, c, l = cart2sph(x, y, z)

    vals = topo.eval(c, l, param='topo')
    elements = triangulate(x, y, z)

    pts = np.array((x, y, z)).T
    write_vtk("topo.vtk", pts, elements, vals, 'topo')

    north_pole = np.array([-4.228])
    south_pole = np.array([-0.056])
    random_point = np.array([0.103])

    np.testing.assert_almost_equal(topo.eval(0, 0, param='topo'), north_pole, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(topo.eval(np.pi, 0, param='topo'), south_pole, decimal=DECIMAL_CLOSE)
    np.testing.assert_almost_equal(topo.eval(np.radians(90 - 53.833333), np.radians(76.500000), param='topo'), random_point, decimal=DECIMAL_CLOSE)