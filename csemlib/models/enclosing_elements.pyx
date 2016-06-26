import numpy as np
cimport numpy as np
import cython

ctypedef np.int_t DTYPE_INT
ctypedef np.float_t DTYPE_FLOAT

@cython.cdivision(True)
@cython.wraparound(False)
@cython.boundscheck(False)
def enclosing_elements(np.ndarray[DTYPE_INT, ndim=1] closest_vertices,
                       np.ndarray[DTYPE_INT, ndim=2] connectivity,
                       np.ndarray[DTYPE_FLOAT, ndim=2] homogeneous_crds,
                       np.ndarray[DTYPE_FLOAT, ndim=1] x_mesh,
                       np.ndarray[DTYPE_FLOAT, ndim=1] y_mesh,
                       np.ndarray[DTYPE_FLOAT, ndim=1] z_mesh):

    # First, invert graph structure.
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] l = np.empty(4, dtype=np.float)
    cdef int vtx_per_elm = 4
    cdef int i, j, k, max_connect, trial_elem
    max_connect = np.amax(np.bincount(connectivity.flatten()))
    cdef np.ndarray[DTYPE_INT, ndim=2] closure = np.full((max_connect, len(x_mesh)), fill_value=-1.0,
                                                         dtype=np.int)
    cdef np.ndarray[DTYPE_INT, ndim=1] bookkeep = np.zeros(len(x_mesh), dtype=np.int)
    cdef int ni = connectivity.shape[0]
    cdef int nj = connectivity.shape[1]
    cdef int nni = len(bookkeep)
    for i in range(ni):
        for j in range(nj):
            closure[bookkeep[connectivity[i,j]],connectivity[i,j]] = i
            bookkeep[connectivity[i,j]] += 1

    # Now, initialize some values.
    cdef np.ndarray[DTYPE_INT, ndim=1] c = np.empty(4, dtype=np.int)
    cdef np.ndarray[DTYPE_FLOAT, ndim=2] t = np.ones((4, 4))
    cdef np.ndarray[DTYPE_FLOAT, ndim=1] barycentric = np.empty(4, np.float)
    cdef num_target_points = homogeneous_crds.shape[0]
    cdef np.ndarray[DTYPE_INT, ndim=2] found_elms = np.empty((4, num_target_points), dtype=np.int)
    cdef np.ndarray[DTYPE_FLOAT, ndim=2] found_bary = np.empty((4, num_target_points), dtype=np.float)

    cdef float vecx, vecy, vecz
    cdef float ab, bb, cb, db, eb, fb, gb, hb, ib
    cdef float ai, bi, ci, di, ei, fi, gi, hi, ii, det


    # Outer loop over all targets.
    for i in range(num_target_points):
        for j in range(max_connect):

            # Get element closure from array.
            trial_elem = closure[j,closest_vertices[i]]

            # Array is sparse, so break if we hit sparsity.
            if trial_elem < 0:
                break

            # Copy xyz coordinates into an array to perform the matrix inversion.
            for k in range(vtx_per_elm):
                t[0, k] = x_mesh[connectivity[trial_elem, k]]
                t[1, k] = y_mesh[connectivity[trial_elem, k]]
                t[2, k] = z_mesh[connectivity[trial_elem, k]]

            # Get barycentric coordinates for this trial element.
            vecx = homogeneous_crds[i, 0] - t[0, 3]
            vecy = homogeneous_crds[i, 1] - t[1, 3]
            vecz = homogeneous_crds[i, 2] - t[2, 3]

            ab = t[0, 0] - t[0, 3]
            db = t[1, 0] - t[1, 3]
            gb = t[2, 0] - t[2, 3]
            bb = t[0, 1] - t[0, 3]
            eb = t[1, 1] - t[1, 3]
            hb = t[2, 1] - t[2, 3]
            cb = t[0, 2] - t[0, 3]
            fb = t[1, 2] - t[1, 3]
            ib = t[2, 2] - t[2, 3]

            det = 1.0 / (( ab * ( eb * ib - fb * hb ) ) - ( bb * ( ib * db - fb * gb ) ) +
                       ( cb * ( db * hb - eb * gb ) ))

            ai = det * (eb * ib - fb * hb)
            bi = det * (db * ib - fb * gb) * (-1)
            ci = det * (db * hb - eb * gb)
            di = det * (bb * ib - cb * hb) * (-1)
            ei = det * (ab * ib - cb * gb)
            fi = det * (ab * hb - bb * gb) * (-1)
            gi = det * (bb * fb - cb * eb)
            hi = det * (ab * fb - cb * db) * (-1)
            ii = det * (ab * eb - bb * db)

            l[0] = ai * vecx + di * vecy + gi * vecz
            l[1] = bi * vecx + ei * vecy + hi * vecz
            l[2] = ci * vecx + fi * vecy + ii * vecz
            l[3] = 1 - l[0] - l[1] - l[2]

            if l[0] >= 0 and l[1] >= 0 and l[2] >= 0 and l[3] >= 0:
                for k in range(vtx_per_elm):
                    found_elms[k, i] = connectivity[trial_elem, k]
                    found_bary[k, i] = l[k]
                break



    return found_elms, found_bary







