import os

import numpy as np
import pytest

import csemlib.background.skeleton as skl
import csemlib.models.crust as crust
import csemlib.models.one_dimensional as m1d
import csemlib.models.s20rts as s20
import csemlib.models.ses3d as s3d

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')

def test_fibonacci_sphere():
    true_y = np.array([-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9])
    true_x = np.array([0.43588989, -0.52658671, 0.0757129, 0.58041368, -0.97977755,
                       0.83952592, -0.24764672, -0.39915719, 0.67080958, -0.40291289])
    true_z = np.array([0., 0.48239656, -0.86270943, 0.75704687, -0.17330885,
                       -0.53403767, 0.92123347, -0.76855288, 0.24497858, 0.16631658])

    points = skl.fibonacci_sphere(10)
    np.testing.assert_allclose(points[0], true_x)
    np.testing.assert_allclose(points[1], true_y)
    np.testing.assert_allclose(points[2], true_z)


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

    np.testing.assert_allclose(m1d.prem_no220(6292, region='upper_mantle'), reg01)
    np.testing.assert_allclose(m1d.prem_no220(6191, region='upper_mantle'), reg02)
    np.testing.assert_allclose(m1d.prem_no220(6051, region='upper_mantle'), reg03)
    np.testing.assert_allclose(m1d.prem_no220(5971, region='upper_mantle'), reg04)
    np.testing.assert_allclose(m1d.prem_no220(5971, region='transition_zone'), reg05)
    np.testing.assert_allclose(m1d.prem_no220(5771, region='transition_zone'), reg06)
    np.testing.assert_allclose(m1d.prem_no220(5701, region='transition_zone'), reg07)
    np.testing.assert_allclose(m1d.prem_no220(5701, region='lower_mantle'), reg08)
    np.testing.assert_allclose(m1d.prem_no220(5600, region='lower_mantle'), reg09)
    np.testing.assert_allclose(m1d.prem_no220(3630, region='lower_mantle'), reg10)
    np.testing.assert_allclose(m1d.prem_no220(3480, region='lower_mantle'), reg11)
    np.testing.assert_allclose(m1d.prem_no220(3480, region='outer_core'), reg12)
    np.testing.assert_allclose(m1d.prem_no220(1221.5, region='outer_core'), reg13)
    np.testing.assert_allclose(m1d.prem_no220(1221.5, region='inner_core'), reg14)

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

    np.testing.assert_allclose(vals_dep, proper_dep)
    np.testing.assert_allclose(vals_vs, proper_vs)


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
    interp = mod.eval(all_cols.ravel(), all_lons.ravel(), all_rads.ravel(),
                      param='dvsv')
    true = mod.data['dvsv'].values.ravel()
    np.testing.assert_allclose(interp, true, atol=1e-2)


def test_s20rts():
    """
    Test to ensure that s20 rts calls returns some proper values.
    :return:
    """

    coords = s3d.Ses3d('japan', os.path.join(TEST_DATA_DIR, 'japan'))
    coords.read()

    mod = s20.S20rts()
    mod.read()

    # c, l, r = np.meshgrid(
    #     coords.data.coords['col'].values,
    #     coords.data.coords['lon'].values,
    #     coords.data.coords['rad'].values[0])

    c, l = np.meshgrid(
        np.linspace(0, np.pi, 180),
        np.linspace(0, 2 * np.pi, 180))

    test = mod.eval(c.ravel(),
                    l.ravel(),
                    [6371],
                    'test').reshape((180, 180))

