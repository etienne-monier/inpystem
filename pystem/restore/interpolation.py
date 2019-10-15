# -*- coding: utf-8 -*-
"""
This module implement the interpolate function which is an interface
between 2D as 3D data and the interpolation function of scipy.
"""

import time

import numpy as np
import scipy.interpolate as spint

from ..tools import sec2str
from ..tools import PCA


def interpolate(
        Y, mask=None, method='nearest', PCA_transform=True, PCA_th='auto',
        verbose=True):
    """Implements data interpolation.

    Three interpolation methods are implemented: nearest neighbor,
    linear interpolation and cubic interpolation.

    Note that cubic interpolation is performed band-by-band in the case
    of 3D data while other methods perform in 3D directly.

    Arguments
    ---------
    Y: (m, n) or (m, n, l) numpy array
        Input data
    mask: optional, None, (m, n) numpy array
        Sampling mask. True for sampled pixel.
        Default is None for full sampling.
    method: optional, 'nearest', 'linear' or 'cubic'
        Interpolation method.
        Default is 'nearest'.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is True.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction.
        Possible values are 'auto' for automatic choice, 'max' for maximum
        value and an int value for user value.
        Default is 'auto'.
    verbose: optional, bool
        Indicates if information text is desired.
        Default is True.

    Returns
    -------
    (m, n) or (m, n, l) numpy array
        Interpolated data.
    float
        Execution time (s).
    """
    # Welcome message
    if verbose:
        print("-- Interpolation reconstruction algorithm --")

    if Y.ndim != 2 and Y.ndim != 3:
        raise ValueError('Invalid data dimension.')

    if mask is None:
        mask = np.ones(Y.shape[:2], dtype=bool)

    if Y.ndim == 3:
        PCA_operator = PCA.PcaHandler(
                Y, mask, PCA_transform=PCA_transform, PCA_th=PCA_th,
                verbose=verbose)
        Y = PCA_operator.Y_PCA

    if method == 'cubic' and Y.ndim == 3:
        # Cubic method only works for 2D data

        dt = time.time()
        Yout = np.zeros(Y.shape)

        for cnt in range(Y.shape[-1]):
            Yout[:, :, cnt] = interpolate(
                Y[:, :, cnt], mask, method=method, verbose=False)

        dt = time.time() - dt

    else:
        # Creates masked data.
        if Y.ndim == 2:
            Ym = Y*mask
        else:
            mask3 = np.tile(mask[:, :, np.newaxis], [1, 1, Y.shape[-1]])
            Ym = Y*mask3

        # This creates NN function data stack.
        nn = Ym.nonzero()
        points = np.asarray(nn).T
        values = Ym[nn]

        if Y.ndim == 3:
            grid_points = np.asarray(
                np.mgrid[0:Y.shape[0], 0:Y.shape[1], 0:Y.shape[2]]
                ).reshape((3, -1)).T
        else:
            grid_points = np.asarray(
                np.mgrid[0:Y.shape[0], 0:Y.shape[1]]
                ).reshape((2, -1)).T

        # Interpolation is performed.
        start = time.time()

        nn_interpolated_values = spint.griddata(
            points, values, grid_points, method=method)

        dt = time.time() - start

        # The interpolated data are reshaped.
        Yout = nn_interpolated_values.reshape(Y.shape)

    if Y.ndim == 3:
        Yout = PCA_operator.inverse(Yout)

    if verbose:
        print(
            "Done in {}.\n---".format(sec2str.sec2str(dt)))

    # Set output info
    InfoOut = {'time': dt}

    if Y.ndim == 3 and PCA_transform:
        PCA_info = {
            'H': PCA_operator.H,
            'PCA_th': PCA_operator.PCA_th,
            'Ym': np.squeeze(PCA_operator.Ym[0, 0, :])
            }
        InfoOut['PCA_info'] = PCA_info

    return Yout, InfoOut
