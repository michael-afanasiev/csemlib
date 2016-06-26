import os

import numpy as np

import csemlib.models.ses3d as s3d


def test_nienke():
    pass

    # path = os.path.join('/Users/mafanasiev/Desktop/europe_1s')
    # mod = s3d.Ses3d('europe', path, components=['dRHO'])
    # mod.read()
    #
    # all_cols, all_lons, all_rads = np.meshgrid(
    #     mod.data.coords['col'].values,
    #     mod.data.coords['lon'].values,
    #     mod.data.coords['rad'].values)
    # interp = mod.eval(all_cols.ravel(), all_lons.ravel(), all_rads.ravel(),
    #                   param=['dRHO'])
    #
    # np.savetxt(os.path.join(path, 'arr.txt'), interp)
