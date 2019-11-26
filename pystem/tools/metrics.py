#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module contains several metric functions.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim


def SNR(xhat, xref):
    """Computes the SNR metric.

    Arguments
    ---------
    xhat: numpy array
        The noised data.
    xref: numpy array
        The noise-free image.

    Returns
    -------
    float
        The SNR value in dB.
    """
    return 10*np.log10(np.mean(xref**2)/np.mean((xref-xhat)**2))


def NMSE(xhat, xref):
    """Computes the normalized mean square metric.

    Arguments
    ---------
    xhat: numpy array
        The noised data.
    xref: numpy array
        The noise-free image.

    Returns
    -------
    float
        The NMSE value.
    """
    return np.linalg.norm(xref - xhat)**2 / np.linalg.norm(xref)**2


def aSAD(xhat, xref):
    """Computes the averaged Spectral Angle Distance metric.

    The input data number of dimensions can be:

    * 1: the data are spectra,
    * 2: the data are matrices of shape (n, M),
    * 3: the data are matrices of shape (m, n, M)

    where M is the spectrum size.

    Arguments
    ---------
    xhat: numpy array
        The noised data.
    xref: numpy array
        The noise-free image.

    Returns
    -------
    float
        The (mean) aSAD value.
    """
    if xref.ndim == 1:
        return float(
            np.arccos(
                np.dot(xhat.T, xref)/(
                    np.linalg.norm(xhat)*np.linalg.norm(xref))))

    elif xref.ndim == 2:
        tmp = np.zeros(xref.shape[0])
        for cnt in range(xref.shape[0]):
            tmp[cnt] = aSAD(xhat=xhat[cnt, :], xref=xref[cnt, :])
        return tmp.mean()

    elif xref.ndim == 3:
        return aSAD(
            xhat=xhat.reshape((-1, xhat.shape[2])),
            xref=xref.reshape((-1, xhat.shape[2])))


def SSIM(xhat, xref):
    """Computes the structural similarity index.

    Arguments
    ---------
    xhat: numpy array
        The noised data.
    xref: numpy array
        The noise-free image.

    Returns
    -------
    float
        The (mean) SSIM value.
    """
    if xref.ndim == 3:
        multichannel = True

    elif xref.ndim == 2:
        multichannel = False

    else:
        raise ValueError('Invalid data number of dimension.')

    return ssim(xref, xhat, multichannel=multichannel)
