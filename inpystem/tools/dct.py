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
    """
    return fp.idctn(a, norm='ortho')


def dct2d_bb(x, shape=None):
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
    shape: optional, (m, n, l) tuple
        This is the data shape. This parameter is required only if input data
        are 2D.

    Returns
    -------
    (l, m*n) or (m, n, l) numpy array
        DCT coefficient matrix.
    """
    if x.ndim == 3:

        return fp.dctn(x, axes=(0, 1), norm='ortho')

    else:

        if shape is None:
            raise ValueError('shape parameter required for 2D data.')

        X = fp.dctn(x.T.reshape(shape), axes=(0, 1), norm='ortho')

        m, n, B = shape
        return X.reshape((m*n, B)).T


def idct2d_bb(a, shape=None):
    """Computes the band-by-band inverse 2D Normalized DCT-II

    If the input a is a 3D data cube, the 2D dct will be computed for each 2D
    images staked along the 2nd axis.

    Arguments
    ---------
    A: (l, m*n) or (m, n, l) numpy array
        2D or 3D multi-band data DCT decomposition.
        If the data has 3 dimensions, the last axis is for spetra.
        If the data is 2D, the first axis is for spectra.
    shape: optional, (m, n, l) tuple
        This is the data shape. This parameter is required only if input data
        are 2D.

    Returns
    -------
    (l, m*n) or (m, n, l) numpy array
        The image matrix.
    """
    # If the data are 2D, it should be transformed back to 3D data.
    if a.ndim == 3:

        return fp.idctn(a, axes=(0, 1), norm='ortho')

    else:

        if shape is None:
            raise ValueError('shape parameter required for 2D data.')

        A = fp.idctn(a.T.reshape(shape), axes=(0, 1), norm='ortho')

        m, n, B = shape
        return A.reshape((m*n, B)).T
