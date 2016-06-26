import numpy as np
import pytest

import csemlib.utils as utl


def test_sph2cart():
    np.testing.assert_allclose(np.array(utl.sph2cart(0, 0, 0)),
                               np.zeros(3))
    np.testing.assert_allclose(np.array(utl.sph2cart(np.math.pi, 0, 0)),
                               np.zeros(3))
    np.testing.assert_almost_equal(np.array(utl.sph2cart(np.math.pi / 2.0, 0)),
                                   np.array((1, 0, 0)))
    np.testing.assert_almost_equal(np.array(utl.sph2cart(np.math.pi / 2.0, np.math.pi / 2.0)),
                                   np.array((0, 1, 0)))
    np.testing.assert_almost_equal(np.array((utl.sph2cart(np.math.pi, np.math.pi / 2.0))),
                                   np.array((0, 0, -1)))
    with pytest.raises(ValueError):
        utl.sph2cart(-1, 0, 0)
    with pytest.raises(ValueError):
        utl.sph2cart(np.math.pi + 1, 0, 0)


def test_cart2sph():
    np.testing.assert_almost_equal(np.array(utl.cart2sph(0, 0, 0)),
                                   np.array((0, 0, 0)))
    np.testing.assert_almost_equal(np.array(utl.cart2sph(1, 0, 0)),
                                   np.array((1, np.math.pi / 2.0, 0)))
    np.testing.assert_almost_equal(np.array(utl.cart2sph(0, 1, 0)),
                                   np.array((1, np.math.pi / 2.0, np.math.pi / 2.0)))
    np.testing.assert_almost_equal(np.array(utl.cart2sph(0, 0, 1)),
                                   np.array((1, 0, 0)))
