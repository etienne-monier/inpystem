# -*- coding: utf-8 -*-
"""This module implements the ITKrMM algorithm.
"""

import time

import numpy as np

import sklearn
from sklearn.feature_extraction import image

from . import sec2str
from ..restore import LS_CLS


def forward_patch_transform(ref, w):
    """Transforms data from 2D/3D array to array whose shape is (w**2, N)
    where w is the patch width and N is the number of patches.

    Arguments
    ---------
    ref: (m, n) or (m, n, l) numpy array
        The input image.
    w: int
        The width (or height) of the patch.

    Returns
    -------
    data: (w**2, N) or (w**2*l, N) numpy array
        The patches stacked version.
        Its shape is (w**2, N) where N is the number of patches if ref is 2D or
        (w**2*l, N) is ref is 3D.
    """
    if ref.ndim == 2:

        data = image.extract_patches_2d(ref, (w, w))

        N, p, _ = data.shape
        return data.reshape((N, p * p)).T

    elif ref.ndim == 3:

        B = ref.shape[2]
        data = image.extract_patches_2d(ref, (w, w))

        N = data.shape[0]

        return data.reshape((N, w * w * B)).T

    else:
        raise ValueError('Invalid number of dimension.')


def inverse_patch_transform(data, shape):
    """Transforms data from array of the form (w**2, N) or (w**2*l, N) where w
    is the patch width, l is the number of bands (in the case of 3D data) and
    N is the number of patches into 2D/3D array.

    Arguments
    ---------
    data: (w**2, N) or (w**2*l, N) numpy array
        The input data.
    shape: (m, n) or (m, n, l)
        The image shape.

    Returns
    -------
    ref: (m, n) or (m, n, l) numpy array
        The input image.
    """
    N = data.shape[1]

    if len(shape) == 2:

        w = int(np.sqrt(data.shape[0]))

        data_r = data.T.reshape((N, w, w))
        return image.reconstruct_from_patches_2d(data_r, shape)

    elif len(shape) == 3:

        B = shape[2]
        w = int(np.sqrt(data.shape[0] / B))

        data_r = data.T.reshape((N, w, w, B))
        return image.reconstruct_from_patches_2d(data_r, shape)

    else:
        raise ValueError('Invalid length for shape.')


def CLS_init(
        Y, Lambda, K=128, S=None, PatchSize=5, mask=None, PCA_transform=False,
        PCA_th='auto', init=None, verbose=True):
    """Dictionary learning initialization based on CLS restoration algorithm.

    Arguments
    ---------
    Y: (m, n, l)n umpy array
        A 3D multi-band image.
    Lambda: float
        Regularization parameter.
    K: optional, int
        The dictionary size.
        Default is 128.
    S: optional, int
        The code sparsity.
        Default is 0.1*PatchSize*l.
    PatchSize: optional, int
        The patch size.
        Default is 5.
    mask: optional, None, (m, n) boolean numpy array
        A sampling mask which is True if the pixel is sampled and False
        otherwise. Default is None for full sampling.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is False as it should be done in dico learning operator.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction. Possible values
        are 'auto' for automatic choice, 'max' for maximum value and an int
        value for user value.
        Default is 'auto'.
    init: optional, None, (m, n, l) numpy array
        The algorithm initialization.
        Default is None for random initialization.
    verbose: optional, bool
        Indicates if information text is desired.
        Default is True.

    Returns
    -------
    (K, l*PatchSize**2) numpy array
        The dictionary for dictionary learning algorithm.
    (K, l*PatchSize**2) numpy array
        The sparse code for dictionary learning algorithm.
    (m, n, l) numpy array
        CLS restored array.
    dict
        Dictionary containing some extra info

    Note
    ----
    Infos in output dictionary:

    * :code:`E` : In the case of partial reconstruction, the cost function
      evolution over iterations.
    * :code:`Gamma` : The array of kept coefficients (order is Fortran-style)
    * :code:`nnz_ratio` : the ratio Gamma.size/(m*n)
    * :code:`H`: the basis of the chosen signal subspace

    """

    if S is None:
        S = 0.1*PatchSize*Y.shape[2]

    #
    # Inpaint using CLS  -------------------
    if verbose:
        print('- CLS reconstruction for init -', end=" ", flush=True)

    t0 = time.time()

    Xhat, InfoOut = LS_CLS.CLS(
        Y, Lambda, mask=mask, PCA_transform=PCA_transform, PCA_th=PCA_th,
        init=init, Nit=None, verbose=False)

    dt = time.time() - t0

    if verbose:
        print('Done in {} -'.format(sec2str.sec2str(dt)))

    #
    # Decompose CLS output into patches  -------------------
    data = forward_patch_transform(Xhat, PatchSize).T

    #
    # Dico learning -------------------
    if verbose:
        print(
            '- Learning the dictionary and getting the code -',
            end=' ', flush=True)

    t0 = time.time()

    dico = sklearn.decomposition.MiniBatchDictionaryLearning(
        n_components=K, transform_n_nonzero_coefs=S, alpha=1, n_iter=500)
    C = dico.fit_transform(data)
    D = dico.components_

    dt = time.time() - t0

    if verbose:
        print('Done in {} -'.format(sec2str.sec2str(dt)))

    return (D, C, Xhat, InfoOut)


def dico_distance(original, new):
    catch_cnt = 0
    total = 0  # Total distance

    d, K = original.shape

    # Make every first atom component be positive.
    new_p = new @ np.diag(np.sign(new[0, :]))

    # I scan all original atom to check it's present in the new dico.
    for atom in original.T:

        atom_p = np.sign(atom)*atom

        distances = np.sum(
            (new_p - np.tile(atom_p[:, np.newaxis], [1, K]))**2, axis=0)

        pos = np.argmin(distances)

        error = 1-np.abs(np.sum(new_p[:, pos]*atom_p))
        total += error
        catch_cnt += error < 1e-2

    ratio = 100*catch_cnt/K
    return ratio, total
