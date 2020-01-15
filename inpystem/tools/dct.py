# -*- coding: utf-8 -*-
"""
This module defines some functions related to DCT decomposition including:

* direct and inverse normalized 2D DCT transform,
* direct and inverse band-by-band DCT transform for multi-band data.
"""

import scipy.fftpack as fp


def dct2d(a):
    """
    Computes the 2D Normalized DCT-II.

    Arguments
    ---------
    X: (m, n) numpy array
        2D image.

    Returns
    -------
    (m, n) numpy array
        DCT coefficient matrix.

    Example
    -------
    >>> from inpystem.tools.dct import dct2d
    >>> import scipy.misc
    >>> face = scipy.misc.face()
    >>> A = dct2d(face.sum(2))
    """
    return fp.dctn(a, norm='ortho')


def idct2d(a):
    """
    Computes the 2D Normalized Inverse DCT-II.

    Arguments
    ---------
    A: (m, n) numpy array
        DCT coefficient matrix

    Returns
    -------
    (m, n) numpy array
        2D image.

    Example
    -------
    >>> from inpystem.tools.dct import dct2d, idct2d
    >>> import scipy.misc
    >>> face = scipy.misc.face()
    >>> A = dct2d(face.sum(2))
    >>> X = idct2d(A)
    """
    return fp.idctn(a, norm='ortho')


def dct2d_bb(x, size=None):
    """
    Computes the band-by-band 2D Normalized DCT-II

    If the input X is a 3D data cube, the 2D dct will be computed for each 2D
    images staked along the 2nd axis.

    Arguments
    ---------
    X: (l, m*n) or (m, n, l) numpy array
        2D or 3D multi-band data.
        If the data has 3 dimensions, the last axis is for spetra.
        If the data is 2D, the first axis is for spectra.
    size: optional, (m, n) tuple
        This is the spatial size. This parameter is required only if input data
        are 2D.

    Returns
    -------
    (l, m*n) or (m, n, l) numpy array
        DCT coefficient matrix.

    Example
    -------
    >>> from inpystem.tools.dct import dct2d_bb
    >>> import scipy.misc
    >>> face = scipy.misc.face()
    >>> A = dct2d_bb(face)
    """

    X = x.copy()

    # If the data are 2D, it should be transformed back to 3D data.
    if X.ndim == 2:

        if size is None:
            raise ValueError(
                'size parameter required for 2D data.')

        B = X.shape[0]
        m, n = size
        X = X.T.reshape((m, n, B))
    else:
        m, n, B = X.shape

    # Perform 2D DCT for all bands.
    for i in range(B):
        X[:, :, i] = dct2d(X[:, :, i])

    # If input data were 2D, reshape output so that to get a 2D matrix.
    if x.ndim == 2:
        X = X.reshape((m*n, B)).T

    return X


def idct2d_bb(a, size=None):
    """Computes the band-by-band inverse 2D Normalized DCT-II

    If the input a is a 3D data cube, the 2D dct will be computed for each 2D
    images staked along the 2nd axis.

    Arguments
    ---------
    A: (l, m*n) or (m, n, l) numpy array
        2D or 3D multi-band data DCT decomposition.
        If the data has 3 dimensions, the last axis is for spetra.
        If the data is 2D, the first axis is for spectra.
    size: optional, (m, n) tuple
        This is the spatial size. This parameter is required only if input data
        are 2D.

    Returns
    -------
    (l, m*n) or (m, n, l) numpy array
        The image matrix.

    Example
    -------
    >>> from inpystem.tools.dct import dct2d_bb, idct2d_bb
    >>> import scipy.misc
    >>> import numpy.testing as npt
    >>> face = scipy.misc.face()
    >>> A = dct2d_bb(face)
    >>> X = idct2d_bb(A)
    >>> npt.assert_allclose(X, face, atol=1e-12)
    """

    A = a.copy()

    # If the data are 2D, it should be transformed back to 3D data.
    if A.ndim == 2:

        if size is None:
            raise ValueError(
                'size parameter required for 2D data.')

        B = A.shape[0]
        m, n = size
        A = A.T.reshape((m, n, B))
    else:
        m, n, B = A.shape

    # Perform 2D DCT for all bands.
    for i in range(B):
        A[:, :, i] = idct2d(A[:, :, i])

    # If input data were 2D, reshape output so that to get a 2D matrix.
    if a.ndim == 2:
        A = A.reshape((m*n, B)).T

    return A
