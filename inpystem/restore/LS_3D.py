# -*- coding: utf-8 -*-
"""This module implements regularized least square restoration methods
adapted to 3D data.

The two methods it gathers are

* **Smoothed SubSpace (3S) algorith**,
* **Smoothed Nuclear Norm (SNN) algorithm**.
"""

import time

import numpy as np
import numpy.linalg as lin
import scipy.sparse as sps

from ..tools import PCA
from ..tools import FISTA
from ..tools import sec2str


def _DX(X):
    """Computes the X finite derivarite along y and x.

    Arguments
    ---------
    X: (m, n, l) numpy array
        The data to derivate.

    Returns
    -------
    tuple
        Tuple of length 2 (Dy(X), Dx(X)).

    Note
    ----
    DX[0] which is derivate along y has shape (m-1, n, l).
    DX[1] which is derivate along x has shape (m, n-1, l).
    """
    return (X[1:, :, :] - X[:-1, :, :],         # D along y
            X[:, 1:, :] - X[:, 0:-1, :])        # D along x


def _Delta(X):
    """Computes the X Laplacian.

    Arguments
    ---------
    X: (m, n, l) numpy array
        The data.

    Returns
    -------
    (m, n, l) numpy array
        The Laplacian of X.
    """
    DY, DX = _DX(X)

    DeltaY = np.pad(
        DY, ((1, 0), (0, 0), (0, 0)), 'constant', constant_values=0) - np.pad(
        DY, ((0, 1), (0, 0), (0, 0)), 'constant', constant_values=0)

    DeltaX = np.pad(
        DX, ((0, 0), (1, 0), (0, 0)), 'constant', constant_values=0) - np.pad(
        DX, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=0)

    return DeltaY + DeltaX


def _f1_SSS(X):
    """Computes the first part of the 3S functional.

    Arguments
    ---------
    X: (m, n, l) numpy array
        The data.

    Returns
    -------
    float
        The first part of the 3S functional.
    """
    DY, DX = _DX(X)
    return lin.norm(DY)**2 + lin.norm(DX)**2


def _proxg_SSS(X, Y, mask, radius):
    """Computes the proximal operator of the 3S constraint.

    Arguments
    ---------
    X: (m, n, l) numpy array
        The variable.
    Y: (m, n, l) numpy array
        The acquired data.
    mask: (m, n) numpy array
        The sampling mask. True means the pixel has been sampled.
    radius: float
        The radius of the constraint sphere.

    Returns
    -------
    (m, n, l) numpy array
        The proximal matrix.
    """

    # Here, we search for the position of non-zero elements of mask.
    #
    # The pos matrix has shape (2, N) where N is the number of mask non-zero
    # entries.
    pos = np.asarray(np.nonzero(mask))
    N = pos.shape[1]

    # The prox operator can be decomposed.
    for c in range(N):

        # Slice object to locate current data.
        pos_s = (np.s_[pos[0, c]], np.s_[pos[1, c]], np.s_[:])

        # The vector is computed and its length evaluated.
        vec = Y[pos_s] - X[pos_s]
        dist = lin.norm(vec)

        # If the variable is outside the sphere, it is projected onto it.
        if dist > radius:
            u = np.negative(vec) / dist
            X[pos_s] = Y[pos_s] + radius * u

    return X


def SSS(Y, Lambda,  mask=None, PCA_transform=True, PCA_th='auto',
        PCA_info=None, scale=1, init=None, Nit=None, verbose=True):
    r"""Smoothed SubSpace algorithm.

    The 3S algorithm denoise or reconstructs a multi-band image possibly
    spatially sub-sampled in the case of spatially smooth images. It is
    well adapted to intermediate scale images.

    This algorithm performs a PCA pre-processing operation to estimate:

    * the data subspace basis :math:`\mathbf{H}`,
    * the subspace dimension :math:`R`,
    * the associated eigenvalues in decreasing order :math:`\mathbf{d}`,
    * the noise level :math:`\hat{\sigma}`.

    After this estimation step, the algorithm solves the folowing
    regularization problem in the PCA space:

    .. math::

        \gdef \S {\mathbf{S}}
        \gdef \Y {\mathbf{Y}}
        \gdef \H {\mathbf{H}}
        \gdef \I {\mathcal{I}}

        \begin{aligned}
        \hat{\S} &= \underset{\S\in\mathbb{R}^{m \times n \times R}}{\arg\min}
                \frac{1}{2R}\left\|\S \mathbf{D}\right\|_\mathrm{F}^2 +
                \frac{\lambda}{2}\sum_{m=1}^{R} w_{m} |\S_{m,:}|_2^2\\
         &\textrm{s.t.}\quad
                \frac{1}{R}|\H_{1:R}^T\Y_{\I(n)}-\S_{\mathcal{I}(n)}|^2_2
                \leq\hat{\sigma}^2,\ \forall n
                \in \{1, \dots,\ m*n\}
        \end{aligned}


    where :math:`\mathbf{Y}` are the corrupted data,  :math:`\mathbf{D}`
    is a spatial finite difference operator and :math:`\mathcal{I}` is
    the set of all sampled pixels.

    Caution
    -------
    It is strongly recomended to perform PCA before running the
    algorithm core. This operation is integrated in this function.

    In case this pre-processing step has already been done, set the
    :code:`PCA_transform` parameter to False to disable the PCA step
    included in the SSS function. If :code:`PCA_transform` is set to
    False, the :code:`PCA_info` parameter is required.

    Arguments
    ---------
    Y (m, n, l) numpy array
        A 3D multi-band image.
    Lambda: float
        Regularization parameter.
    mask: optional, None, (m, n) numpy array
        A sampling mask which is True if the pixel is sampled.
        Default is None for full sampling.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is True.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction.
        Possible values are 'auto' for automatic choice, 'max' for maximum
        value and an int value for user value.
        Default is 'auto'.
    PCA_info: optional, dict
        In case PCA_transform is False, some extra info should be given
        to SSS.
        The required keys for PCA_info are:

            * 'd' which are the PCA eigenvalues.
            * 'sigma' which is an estimate of the data noise std.
    scale: optional, float
        Scales the prox operator sphere radius. Should lay in ]0, +inf[.
        Default is 1.
    init: optional, None, (m, n, l) numpy array
        The algorithm initialization.
        Default is None for random initialization.
    Nit: optional, None, int
        Number of iteration in case of inpainting. If None, the iterations
        will stop as soon as the functional no longer evolve.
        Default is None.
    verbose: optional, bool
        Indicates if information text is desired.
        Default is True.

    Returns
    -------
    (m, n, l) numpy array
        The reconstructed/denoised multi-band image.
    dict
        A dictionary containing some extra info

    Note
    ----
    Infos in output dictionary:

    * :code:`E`: in the case of partial reconstruction, the cost function
      evolution over iterations.
    * :code:`H` : the basis of the chosen signal subspace


    References
    ----------
    Monier, E., Oberlin, T., Brun, N., Tencé, M., de Frutos, M., &
    Dobigeon, N. (2018). Reconstruction of Partially Sampled Multiband
    Images—Application to STEM-EELS Imaging. IEEE Trans. Comput. Imag.,
    4(4), 585–598.

    """

    # Test and initializations
    if (Lambda < 0):
        raise ValueError('Lambda parameter is not positive.')
    if scale <= 0:
        raise ValueError('Input scale should be strict. positive.')

    if mask is None:
        mask = np.ones(Y.shape[:2])

    if init is None:
        init = np.random.randn(*Y.shape)

    # Welcome message
    if verbose:
        print("-- 3S Reconstruction algorithm --")

    #
    # Dimension reduction
    #

    PCA_operator = PCA.PcaHandler(
        Y, mask, PCA_transform=PCA_transform, PCA_th=PCA_th, verbose=verbose)
    Y_PCA, PCA_th = PCA_operator.Y_PCA, PCA_operator.PCA_th

    init = PCA_operator.direct(init)

    if PCA_operator.InfoOut is None:
        d, sigma = PCA_info['d'], PCA_info['sigma']
    else:
        d, sigma = PCA_operator.InfoOut['d'], PCA_operator.InfoOut['sigma']

    #
    # Center and normalize data
    #
    Y_m, Y_std = Y_PCA.mean(), Y_PCA.std()
    init_m, init_std = init.mean(), init.std()

    Y_PCA = (Y_PCA - Y_m)/Y_std
    init = (init - init_m)/init_std

    #
    # Get dimensions and start timing
    #

    m, n, _ = Y_PCA.shape
    P = m*n

    start = time.time()

    #
    # Definition of FISTA functions
    #

    # f function
    epsilon = 5e-6
    w = sigma**2 / np.maximum(d[:PCA_th] - sigma**2, epsilon)
    W = sps.diags(w, 0)

    L = 8 / PCA_th + Lambda * np.max(w)

    #
    # FISTA solver
    #
    solver = FISTA.FISTA(
        #
        f=lambda X: _f1_SSS(X) / (2 * PCA_th) + Lambda * w.T.dot(
            lin.norm(X.reshape((P, -1)).T, axis=1)**2),
        #
        df=lambda X: _Delta(X) / PCA_th + Lambda * W.dot(
            X.reshape((P, -1)).T).T.reshape(Y_PCA.shape),
        #
        L=L,
        #
        g=lambda X: 0,
        #
        pg=lambda X: _proxg_SSS(
            X, Y_PCA, mask, np.sqrt(PCA_th) * sigma * scale),
        #
        shape=Y_PCA.shape,
        init=init,
        Nit=Nit,
        verbose=verbose)

    X_PCA, InfoOut_FISTA = solver.execute()

    # Catch output info
    InfoOut = {'E': InfoOut_FISTA['E'],
               'time': time.time() - start}
    if PCA_transform:
        PCA_info = {
            'H': PCA_operator.H,
            'PCA_th': PCA_operator.PCA_th,
            'Ym': np.squeeze(PCA_operator.Ym[0, 0, :])
            }
        InfoOut['PCA_info'] = PCA_info

    # Output managing.
    #

    X_PCA = X_PCA * Y_std + Y_m
    Xhat = PCA_operator.inverse(X_PCA)

    if (verbose):
        print("Done in {}.\n---".format(
            sec2str.sec2str(time.time()-start)))

    return Xhat, InfoOut


def _soft_thresholding(a, t):
    """Soft thresholding operator.

    Arguments
    ---------
    a: (N, ) numpy array
        Input array
    t: float
        Threshold.

    Returns
    -------
    (N, ) numpy array
        Thresholded array
    """
    return np.sign(a) * np.maximum(np.abs(a) - t, 0)


def _proxg_SNN(X, Mu):
    """
    """

    # SVD decomposition
    m, n, B = X.shape
    U, s, Vh = np.linalg.svd(X.reshape(m*n, B).T, full_matrices=False)

    # Soft-thresholding the singular values.
    s_bar = _soft_thresholding(s, Mu)

    # Prox output
    prox = U @ np.diag(s_bar) @ Vh

    return prox.T.reshape((m, n, B))


def _g_SNN(X, Mu):
    """
    """
    m, n, B = X.shape
    X_tmp = X.reshape((m*n, B)).T
    return Mu * np.linalg.norm(X_tmp, 'nuc')


def SNN(Y, Lambda, Mu,  mask=None, PCA_transform=True, PCA_th='auto',
        init=None, Nit=None, verbose=True):
    r"""Smoothed Nuclear Norm algorithm.

    The SNN algorithm denoise or reconstructs a multi-band image possibly
    spatially sub-sampled in the case of spatially smooth images. It is
    well adapted to intermediate scale images.

    This algorithm solves the folowing optimization problem:

    .. math::

        \gdef \X {\mathbf{X}}
        \gdef \Y {\mathbf{Y}}
        \gdef \H {\mathbf{H}}
        \gdef \I {\mathcal{I}}

        \hat{\X} = \underset{\X\in\mathbb{R}^{m \times n \times B}}{\arg\min}
            \frac{1}{2}||\Y_\I - \X_\I||_\mathrm{F}^2 +
            \frac{\lambda}{2}\left\|\X \mathbf{D}\right\|_\mathrm{F}^2 +
            \mu ||\X||_*

    where :math:`\mathbf{Y}` are the corrupted data,  :math:`\mathbf{D}`
    is a spatial finite difference operator and :math:`\mathcal{I}` is
    the set of all sampled pixels.

    This algorithm can perform a PCA pre-processing operation to estimate:

    * the data subspace basis :math:`\mathbf{H}`,
    * the subspace dimension :math:`R`.

    This is particularly usefull to reduce the data dimension and the
    execution time and to impose a data low-rank property.

    Caution
    -------
    It is strongly recomended to perform PCA before running the
    algorithm core. This operation is integrated in this function.

    In case this pre-processing step has already been done, set the
    :code:`PCA_transform` parameter to False to disable the PCA step
    included in the CLS function.

    Arguments
    ---------
    Y (m, n, l) numpy array
        A 3D multi-band image.
    Lambda: float
        Regularization parameter #1.
    Mu: float
        Regularization parameter #2.
    mask: optional, None, (m, n) numpy array
        A sampling mask which is True if the pixel is sampled.
        Default is None for full sampling.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is True.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction.
        Possible values are 'auto' for automatic choice, 'max' for maximum
        value and an int value for user value.
        Default is 'auto'.
    init: optional, None, (m, n, l) numpy array
        The algorithm initialization.
        Default is None for random initialization.
    Nit: optional, None, int
        Number of iteration in case of inpainting. If None, the iterations
        will stop as soon as the functional no longer evolve.
        Default is None.
    verbose: optional, bool
        Indicates if information text is desired.
        Default is True.

    Returns
    -------
    (m, n, l) numpy array
        The reconstructed/denoised multi-band image.
    dict
        A dictionary containing some extra info

    Note
    ----
    Infos in output dictionary:

    * :code:`E`: in the case of partial reconstruction, the cost function
      evolution over iterations.
    * :code:`H` : the basis of the chosen signal subspace

    References
    ----------
    Monier, E., Oberlin, T., Brun, N., Tencé, M., de Frutos, M., &
    Dobigeon, N. (2018). Reconstruction of Partially Sampled Multiband
    Images—Application to STEM-EELS Imaging. IEEE Trans. Comput. Imag.,
    4(4), 585–598.

    """

    # Test and initializations
    if (Lambda < 0):
        raise ValueError('Lambda parameter is not positive.')
    if (Mu < 0):
        raise ValueError('Lambda parameter is not positive.')

    if mask is None:
        mask = np.ones(Y.shape[:2])

    if init is None:
        init = np.random.randn(*Y.shape)

    # Welcome message
    if verbose:
        print("-- SNN Reconstruction algorithm --")

    #
    # Dimension reduction
    #

    PCA_operator = PCA.PcaHandler(
        Y, mask, PCA_transform=PCA_transform, PCA_th=PCA_th, verbose=verbose)
    Y_PCA = PCA_operator.Y_PCA

    init = PCA_operator.direct(init)

    #
    # Center and normalize data
    #
    Y_m, Y_std = Y_PCA.mean(), Y_PCA.std()
    init_m, init_std = init.mean(), init.std()

    Y_PCA = (Y_PCA - Y_m)/Y_std
    init = (init - init_m)/init_std

    #
    # Get dimensions and start timing
    #

    m, n, _ = Y_PCA.shape

    start = time.time()

    #
    # Definition of FISTA functions
    #

    # f function
    mask3 = np.tile(mask[:, :, np.newaxis], [1, 1, Y_PCA.shape[-1]])
    L = 1 + Lambda * 8

    #
    # FISTA solver
    #
    solver = FISTA.FISTA(
        #
        f=lambda X:
            0.5 * lin.norm((Y_PCA - X) * mask3)**2 +
            Lambda * _f1_SSS(X) / 2,
        #
        df=lambda X: (X - Y_PCA) * mask3 + Lambda * _Delta(X),
        #
        L=L,
        #
        g=lambda X: _g_SNN(X, Mu),
        #
        pg=lambda X: _proxg_SNN(X, Mu),
        #
        shape=Y_PCA.shape,
        init=init,
        Nit=Nit,
        verbose=verbose)

    X_PCA, InfoOut_FISTA = solver.execute()

    # Catch output info
    InfoOut = {'E': InfoOut_FISTA['E'],
               'time': time.time() - start}
    if PCA_transform:
        PCA_info = {
            'H': PCA_operator.H,
            'PCA_th': PCA_operator.PCA_th,
            'Ym': np.squeeze(PCA_operator.Ym[0, 0, :])
            }
        InfoOut['PCA_info'] = PCA_info

    # Output managing.
    #
    X_PCA = (X_PCA * Y_std) + Y_m
    Xhat = PCA_operator.inverse(X_PCA)

    if (verbose):
        print("Done in {}.\n---".format(
            sec2str.sec2str(time.time() - start)))

    return Xhat, InfoOut
