import numpy as np

from csemlib.background.skeleton import multiple_fibonacci_spheres, multiple_fibonacci_resolution
from csemlib.models import crust
from csemlib.models.model import triangulate
from csemlib.models.model import write_vtk
from csemlib.models.s20rts import S20rts
from csemlib.utils import sph2cart, cart2sph


def prem_no220(rad, region=None):
    regions = {'upper_mantle', 'transition_zone', 'lower_mantle', 'outer_core',
               'inner_core'}
    if region:
        region = region.lower()
        if region not in regions:
            raise ValueError('region must be one of: {}.'.format(
                ', '.join(regions)))

    r_earth = 6371.0
    s_ani = 0.0011
    r_ani = 6201.0
    x = rad / r_earth

    if rad >= 6292 and region == 'upper_mantle':
        rho = 2.6910 + 0.6924 * x
        vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        return rho, vpv, vsv, vsh

    elif 6292 >= rad >= 6191 and region == 'upper_mantle':
        rho = 2.6910 + 0.6924 * x
        vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        return rho, vpv, vsv, vsh

    elif 6191 >= rad >= 6051 and region == 'upper_mantle':
        rho = 9.1790 - 5.9841 * x
        vpv = 40.5988 - 33.5317 * x
        vsv = vsh = 16.8261 - 12.7527 * x
        return rho, vpv, vsv, vsh

    elif 6051 >= rad >= 5971 and region == 'upper_mantle':
        rho = 7.1089 - 3.8045 * x
        vpv = 20.3926 - 12.2569 * x
        vsv = vsh = 8.9496 - 4.4597 * x
        return rho, vpv, vsv, vsh

    elif 5971 >= rad >= 5771 and region == 'transition_zone':
        rho = 11.2494 - 8.0298 * x
        vpv = 39.7027 - 32.6166 * x
        vsv = vsh = 22.3512 - 18.5856 * x
        return rho, vpv, vsv, vsh

    elif 5771 >= rad >= 5701 and region == 'transition_zone':
        rho = 5.3197 - 1.4836 * x
        vpv = 19.0957 - 9.8672 * x
        vsv = vsh = 9.9839 - 4.9324 * x
        return rho, vpv, vsv, vsh

    elif 5701 >= rad >= 5600 and region == 'lower_mantle':
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vpv = 29.2766 - 23.6026 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 22.3459 - 17.2473 * x - 2.0834 * x * x + 0.9783 * x * x * x
        return rho, vpv, vsv, vsh

    elif 5600 >= rad >= 3630 and region == 'lower_mantle':
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vpv = 24.9520 - 40.4673 * x + 51.4832 * x * x - 26.6419 * x * x * x
        vsv = vsh = 11.1671 - 13.7818 * x + 17.4575 * x * x - 9.2777 * x * x * x
        return rho, vpv, vsv, vsh

    elif 3630 >= rad >= 3480 and region == 'lower_mantle':
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vpv = 15.3891 - 5.3181 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 6.9254 + 1.4672 * x - 2.0834 * x * x + 0.9783 * x * x * x
        return rho, vpv, vsv, vsh

    elif 3480 >= rad >= 1221.5 and region == 'outer_core':
        rho = 12.5815 - 1.2638 * x - 3.6426 * x * x - 5.5281 * x * x * x
        vpv = 11.0487 - 4.0362 * x + 4.8023 * x * x - 13.5732 * x * x * x
        vsv = vsh = 0.0
        return rho, vpv, vsv, vsh

    elif rad <= 1221.5 and region == 'inner_core':
        rho = 13.0885 - 8.8381 * x * x
        vpv = 11.2622 - 6.3640 * x * x
        vsv = vsh = 3.6678 - 4.4475 * x * x
        return rho, vpv, vsv, vsh

    else:
        raise ValueError(
            'Radius of {} could not be processed. Ensure correct region is '
            'specified ({})'.format(rad, ', '.join(regions)))


def prem_no220_no_regions(rad):
    """
    :param rad: distance from core in km
    :return rho, vpv, vsv, vsh:
    """
    r_earth = 6371.0
    s_ani = 0.0011
    r_ani = 6201.0
    x = rad / r_earth

    if rad >= 6292:
        rho = 2.6910 + 0.6924 * x
        vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        return rho, vpv, vsv, vsh

    elif 6292 > rad >= 6191:
        rho = 2.6910 + 0.6924 * x
        vpv = 4.1875 + 3.9382 * x
        vsv = 2.1519 + 2.3481 * x
        vsh = vsv + s_ani * (rad - r_ani)
        return rho, vpv, vsv, vsh

    elif 6191 > rad >= 6051:
        rho = 9.1790 - 5.9841 * x
        vpv = 40.5988 - 33.5317 * x
        vsv = vsh = 16.8261 - 12.7527 * x
        return rho, vpv, vsv, vsh

    elif 6051 > rad >= 5971:
        rho = 7.1089 - 3.8045 * x
        vpv = 20.3926 - 12.2569 * x
        vsv = vsh = 8.9496 - 4.4597 * x
        return rho, vpv, vsv, vsh

    elif 5971 > rad >= 5771:
        rho = 11.2494 - 8.0298 * x
        vpv = 39.7027 - 32.6166 * x
        vsv = vsh = 22.3512 - 18.5856 * x
        return rho, vpv, vsv, vsh

    elif 5771 > rad >= 5701:
        rho = 5.3197 - 1.4836 * x
        vpv = 19.0957 - 9.8672 * x
        vsv = vsh = 9.9839 - 4.9324 * x
        return rho, vpv, vsv, vsh

    elif 5701 > rad >= 5600:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vpv = 29.2766 - 23.6026 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 22.3459 - 17.2473 * x - 2.0834 * x * x + 0.9783 * x * x * x
        return rho, vpv, vsv, vsh

    elif 5600 > rad >= 3630:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vpv = 24.9520 - 40.4673 * x + 51.4832 * x * x - 26.6419 * x * x * x
        vsv = vsh = 11.1671 - 13.7818 * x + 17.4575 * x * x - 9.2777 * x * x * x
        return rho, vpv, vsv, vsh

    elif 3630 > rad >= 3480:
        rho = 7.9565 - 6.4761 * x + 5.5283 * x * x - 3.0807 * x * x * x
        vpv = 15.3891 - 5.3181 * x + 5.5242 * x * x - 2.5514 * x * x * x
        vsv = vsh = 6.9254 + 1.4672 * x - 2.0834 * x * x + 0.9783 * x * x * x
        return rho, vpv, vsv, vsh

    elif 3480 > rad >= 1221.5:
        rho = 12.5815 - 1.2638 * x - 3.6426 * x * x - 5.5281 * x * x * x
        vpv = 11.0487 - 4.0362 * x + 4.8023 * x * x - 13.5732 * x * x * x
        vsv = vsh = 0.0
        return rho, vpv, vsv, vsh

    elif 1221.5 > rad >= 0:
        rho = 13.0885 - 8.8381 * x * x
        vpv = 11.2622 - 6.3640 * x * x
        vsv = vsh = 3.6678 - 4.4475 * x * x
        return rho, vpv, vsv, vsh

    # This should never happen in theory
    elif rad < 0:
        raise ValueError('Negative radius specified for 1D_prem')

def prem_eval_point_cloud(rad):
    g = np.vectorize(prem_no220_no_regions)
    rho, vpv, vsv, vsh = g(rad)
    return rho, vpv, vsv, vsh

def test_add_crust_and_s20rts_prem():
    # Generate point cloud
    num_layers = 5
    radii = np.linspace(6371.0, 0.0, num_layers)
    r_earth = 6371.0
    res = r_earth / num_layers

    x, y, z = multiple_fibonacci_resolution(radii, resolution=res, min_samples=10)
    #x, y, z = multiple_fibonacci_spheres(radii, n_samples, normalized_radius=False)
    r, c, l = cart2sph(x, y, z)

    # Evaluate Prem
    rho, vpv, vsv, vsh = prem_eval_point_cloud(r)
    pts = np.array((c, l, r, vsv))

    # Evaluate s20rts
    s20mod = S20rts()
    s20mod.read()
    pts = s20mod.eval_point_cloud_non_norm(*pts)

    cst = crust.Crust()
    pts = cst.eval_point_cloud(*pts.T)

    # Generate mesh for plotting (normalised coordinates)
    x, y, z = sph2cart(pts[:, 0], pts[:, 1], pts[:, 2]/ r_earth)
    elements = triangulate(x, y, z)

    # Write to vtk
    coords = np.array((x, y, z)).T

    write_vtk("crust_vsv.vtk", coords, elements, pts[:, 3], 'vsv')

test_add_crust_and_s20rts_prem()
