#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements the BPFA algorithm.
"""

from pathlib import Path

import numpy as np

from ..tools import PCA
from ..tools import sec2str
from ..tools import matlab_interface as matlab


def BPFA_matlab(Y, mask, PatchSize=5, Omega=1, K=128, Nit=100, step=1,
                PCA_transform=True, PCA_th='auto', verbose=True):
    """Implements BPFA algorithm for python.

    This function does not properly executes BPFA but it calls
    the Matlab BPFA code.

    Arguments
    ---------
    Y: (m, n) or (m, n, l) numpy array
        The input data.
    mask: (m, n) numpy array
        The acquisition mask.
    PatchSize: int
        The patch width.
        Default is 5.
    Omega: int
        The Omega parameter.
        Default is 1.
    K: int
        The dictionary dimension.
        Default is 128.
    Nit: int
        The number of iterations.
        Default is 100.
    step: int
        The distance between two consecutive patches.
        Default is 1 for full overlap.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is True.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction.
        Possible values are 'auto' for automatic choice, 'max' for maximum
        value and an int value for user value.
        Default is 'auto'.
    verbose: bool
        The verbose parameter.
        Default is True.

    Returns
    -------
    (m, n) or (m, n, l) numpy array
        Restored data.
    dict
        Aditional informations. See Notes.

    Note
    ----
    The output information keys are:

    * :code:`time`: Execution time in seconds.
    * :code:`Z`
    * :code:`A`
    * :code:`S`
    """
    if Omega <= 0:
        raise ValueError('The Omega parameter should be positive.')

    if verbose:
        print("-- BPFA reconstruction algorithm --")

    #
    # Dimension reduction
    #
    if Y.ndim == 3:
        PCA_operator = PCA.PcaHandler(
            Y, mask, PCA_transform=PCA_transform,
            PCA_th=PCA_th, verbose=verbose)
        Y_PCA = PCA_operator.Y_PCA
    else:
        Y_PCA = Y.copy()
    #
    # Center and normalize data
    #
    Y_m, Y_std = Y_PCA.mean(), Y_PCA.std()
    Y_PCA = (Y_PCA - Y_m)/Y_std

    # Execute program.
    #

    # Arguments.
    Dico = {'Y': Y_PCA,
            'mask': mask,
            'PatchSize': PatchSize,
            'Omega': Omega,
            'K': K,
            'iter': Nit,
            'Step': step,
            'verbose': verbose}

    # Executes program
    data = matlab.matlab_interface(
        Path(__file__).parent / 'MatlabCodes' / 'BPFA' /
        'BPFA_for_python.m',
        Dico)

    # Get output data
    X_PCA = data['Xhat']
    dt = data['time']
    dico_hat = data['A']
    # Z = data['Z']
    # S = data['S']

    # Output managing.
    #

    # Create output dico
    # Reshape output dico
    shape_dico = (K, PatchSize, PatchSize) if Y.ndim == 2 else (
        K, PatchSize, PatchSize, Y_PCA.shape[-1])
    dico = dico_hat.T.reshape(shape_dico)

    # Create output data
    X_PCA = X_PCA * Y_std + Y_m
    if Y.ndim == 3:
        Xhat = PCA_operator.inverse(X_PCA)
    else:
        Xhat = X_PCA

    # Create InfoOut
    InfoOut = {'time': dt, 'dico': dico}

    if Y.ndim == 3:
        PCA_info = {
            'H': PCA_operator.H,
            'PCA_th': PCA_operator.PCA_th,
            'Ym': np.squeeze(PCA_operator.Ym[0, 0, :])
            }
        InfoOut['PCA_info'] = PCA_info

    if verbose:
        print(
            "Done in {}.\n---".format(sec2str.sec2str(dt)))

    return Xhat, InfoOut
