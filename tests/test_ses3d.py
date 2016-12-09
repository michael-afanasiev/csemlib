import os

import numpy as np
import csemlib.background.skeleton as skl

import csemlib.models.ses3d as s3d
from csemlib.models.model import triangulate, write_vtk


TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'test_data')
VTK_DIR = os.path.join(os.path.split(__file__)[0], 'vtk')
DECIMAL_CLOSE = 3




def fail_ses3d():
    mod = s3d.Ses3d('japan', os.path.join(TEST_DATA_DIR, 'japan'),
                    components=['drho', 'dvsv', 'dvsh', 'dvp'])
    mod.read()

    # Generate Fibonacci sphere at 20 km depth
    x, y, z = skl.fibonacci_sphere(100000)
    x *= 6250.0
    y *= 6250.0
    z *= 6250.0

    # Eval ses3d
    interp = mod.eval(x, y, z, param=['dvsv', 'drho', 'dvsh', 'dvp'])

    # Write to vtk
    elements = triangulate(x, y, z)
    pts = np.array((x, y, z)).T
    write_vtk(os.path.join(VTK_DIR, 'ses3d_fail.vtk'), pts, elements, interp[:,0], 'ses3d')

fail_ses3d()