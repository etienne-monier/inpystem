# -*- coding: utf-8 -*-
"""This module implements regularized least square restoration methods
adapted to 3D data.

The two methods it gathers are

* **Cosine Least Square (CLS) algorith**,
* **Post-LS Cosine Least Square (Post_LS_CLS) algorithm**.
"""

import time

import numpy as np
import numpy.linalg as lin

from ..tools import PCA
from ..tools import FISTA
from ..tools import dct
from ..tools import sec2str


def _proxg_cls(X, Lambda):
    """Implementation of the proximal operator of g(X)=Lambda*||X*Psi||_{2,1}
    where Psi is the band-by-band DCT transform.

    Arguments
    ---------
    X: (m, n, l) numpy array
        The data matrix.
    Lambda: float
        The Lambda parameter.

    Returns
    -------
    (m, n, l) numpy array
        Proximal operator of g(X).
    float
        Percentage of non-zero pgX DCT coefficients.
    numpy array
        List of (flattened) non-zero pgX DCT coefficients indexes.

    """
    # Get shape and DCT transform
    m, n, B = X.shape
    A = dct.dct2d_bb(X)

    # Vector containing the l2 norms of the DCT(X) spectra.
    Normed = lin.norm(A, ord=2, axis=2)

    # Repeat Normed along spectrum axis.
    NormedR = np.repeat(Normed[:, :, np.newaxis], B, axis=2)

    # Indices of the spectra that should not be 0 after thresholding.
    Gamma = np.flatnonzero(Normed > Lambda)
    nnz = np.nonzero(NormedR > Lambda)

    # Create output matrix
    At = np.zeros(A.shape)

    # Thresholding
    At[nnz] = (1 - Lambda / NormedR[nnz]) * A[nnz]

    # Inverse DCT transformation
    pgX = dct.idct2d_bb(At)

    return (pgX, Gamma.size / (n * m), Gamma)


def CLS(Y, Lambda, mask=None, PCA_transform=True, PCA_th='auto', init=None,
        Nit=None, verbose=True):
    r"""Cosine Least Square algorithm

    The CLS algorithm denoises or reconstructs a multi-band image possibly
    spatially sub-sampled in the case of spatially sparse content in the DCT
    basis. It is well adapted to periodic data.

    This algorithm solves the folowing optimization problem:

    .. math::

        \gdef \X {\mathbf{X}}
        \gdef \Y {\mathbf{Y}}
        \gdef \H {\mathbf{H}}
        \gdef \I {\mathcal{I}}

        \hat{\X} = \underset{\X\in\mathbb{R}^{m \times n \times B}}{\arg\min}
            \frac{1}{2}||\Y_\I - \X_\I||_\mathrm{F}^2 +
            \lambda ||\X \Psi||_{2, 1}


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
    **PCA_transform** parameter to False to disable the PCA step
    included in the SSS function. If PCA_transform is set to False, the
    PCA_info parameter is required.

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

    * :code:`E` : In the case of partial reconstruction, the cost function
      evolution over iterations.
    * :code:`Gamma` : The array of kept coefficients (order is Fortran-style)
    * :code:`nnz_ratio` : the ratio Gamma.size/(m*n)
    * :code:`H`: the basis of the chosen signal subspace
    """

    # Test and initializations
    if (Lambda < 0):
        raise ValueError('Lambda parameter is not positive.')

    if mask is None:
        mask = np.ones(Y.shape[:2])

    if init is None:
        init = np.random.randn(*Y.shape)

    # Welcome message
    if verbose:
        print("-- CLS Reconstruction algorithm --")

    # Dimension reduction
    PCA_operator = PCA.PcaHandler(
        Y, mask, PCA_transform=PCA_transform, PCA_th=PCA_th, verbose=verbose)
    Y_PCA, PCA_th = PCA_operator.Y_PCA, PCA_operator.PCA_th

    init = PCA_operator.direct(init)

    #
    # Center and normalize data
    #
    Y_m, Y_std = Y_PCA.mean(), Y_PCA.std()
    init_m, init_std = init.mean(), init.std()

    Y_PCA = (Y_PCA - Y_m)/Y_std
    init = (init - init_m)/init_std

    #
    # Separates denoising vs. inpainting
    #
    m, n = Y_PCA.shape[:2]
    N = mask.sum()
    P = m*n

    start = time.time()

    if (N == P):

        #
        # Denoising
        #
        # In this case, the procedure consists in simply applying the g prox
        # operator to Y_PCA.
        #

        X_PCA, nnz_ratio, Gamma = _proxg_cls(Y_PCA, Lambda)

        # As the mean of Xwm had been removed before thresholding, the coeff
        # 0 may have been removed from I.
        # Let's put it back, if removed.

        # This code has been removed as the data are centered before
        # processing.
        # if not np.isin(0,Gamma):
        #     Gamma = np.insert(Gamma,0,0)
        #     KeptRatio = Gamma.size/P

        localInfo = {}

    else:

        #
        # Inpainting
        #

        mask3 = np.tile(mask[:, :, np.newaxis], [1, 1, PCA_th])
        L = 1
        # import ipdb; ipdb.set_trace()
        #
        # FISTA solver
        #
        solver = FISTA.FISTA(
            #
            f=lambda X: 1 / 2 * lin.norm(((Y_PCA - X)*mask3).flatten())**2,
            #
            df=lambda X: (X - Y_PCA)*mask3,
            #
            L=L,
            #
            g=lambda X: Lambda * np.sum(
                lin.norm(dct.dct2d_bb(X), 2, axis=2)),
            #
            pg=lambda X: _proxg_cls(X, Lambda / L)[0],
            #
            shape=Y_PCA.shape,
            init=init,
            Nit=Nit,
            verbose=verbose)

        X_PCA, InfoOut_FISTA = solver.execute()

        # Get extra info
        _, nnz_ratio, Gamma = _proxg_cls(X_PCA, 1e-10)
        # Lambda can be whatever, it does not affect X_PCA.
        # The Lambda parameter here is the level above which a coeff
        # is no more considered to be zero.
        # This level should not be 0 exactly as machine non-zero can
        # appear when performing direct, then inverse DCT.

        localInfo = {'E': InfoOut_FISTA['E']}

    #
    # Output managing.
    #
    X_PCA = (X_PCA * Y_std) + Y_m
    Xhat = PCA_operator.inverse(X_PCA)

    dt = time.time() - start
    commonInfo = {'Gamma': Gamma,
                  'nnz_ratio': nnz_ratio,
                  'time': dt}

    if PCA_transform:
        PCA_info = {
            'H': PCA_operator.H,
            'PCA_th': PCA_operator.PCA_th,
            'Ym': np.squeeze(PCA_operator.Ym[0, 0, :])
            }
        commonInfo['PCA_info'] = PCA_info

    InfoOut = {**localInfo, **commonInfo}

    if (verbose):
        print("""Final ratio of nonzero coefficients is {}.
{} nonzero coefficients over {}.
Done in {}.
--
""".format(nnz_ratio, Gamma.size, P, sec2str.sec2str(dt)))

    return Xhat, InfoOut


def _proxg_refitting(A, Gamma):
    """Sets all pixels not in Gamma to 0.

    Arguments
    ---------
    A: (m, n, l) numpy array
        3D input image.
    Gamma: numpy array
        Thresholded pixels indexes.

    Returns
    -------
    (m, n, l) numpy array
        Thresholded array.
    """
    # Output thresholded data.
    At = np.zeros(A.shape)

    # Only spectra whose index is in Gamma are copied.
    i_arr, j_arr = np.unravel_index(Gamma, A.shape[:2])
    At[i_arr, j_arr, :] = A[i_arr, j_arr, :]

    return At


def Post_LS_CLS(Y, Lambda, mask=None, PCA_transform=True, PCA_th='auto',
                init=None, Nit=None, verbose=True):
    """Post-Lasso CLS algorithm.

    This algorithms consists in applying CLS to restore the data and
    determine the data support in DCT basis. A post-least square
    optimization is performed to reduce the coefficients bias.

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
    tuple
        A 2-tuple whose alements are the CLS and reffitting information
        dictionaries.

    Note
    ----
    Infos in output dictionary:

    * :code:`E_CLS` : In the case of partial reconstruction, the cost
      function evolution over iterations.
    * :code:`E_post_ls` : In the case of partial reconstruction, the
      cost function evolution over iterations.
    * :code:`Gamma` : The array of kept coefficients
      (order is Fortran-style)
    * :code:`nnz_ratio` : the ratio Gamma.size/(m*n)
    * :code:`H`: the basis of the chosen signal subspace
    """

    # Welcome message
    if verbose:
        print("Post-Lasso CLS Reconstruction algorithm...")

    #
    # Dimension reduction
    #

    PCA_operator = PCA.PcaHandler(
        Y, mask, PCA_transform=PCA_transform, PCA_th=PCA_th, verbose=verbose)
    Y_PCA, PCA_th = PCA_operator.Y_PCA, PCA_operator.PCA_th

    init = PCA_operator.direct(init)

    #
    # Center and normalize data
    #
    Y_m, Y_std = Y_PCA.mean(), Y_PCA.std()
    init_m, init_std = init.mean(), init.std()

    Y_PCA = (Y_PCA - Y_m)/Y_std
    init = (init - init_m)/init_std

    #
    # CLS reconstruction
    #

    Xhat_PCA, InfoOut_CLS = CLS(
        Y_PCA, Lambda, mask=mask, PCA_transform=False, PCA_th=PCA_th,
        init=init, Nit=Nit, verbose=verbose)

    #
    # Refitting
    #

    Gamma = InfoOut_CLS['Gamma']
    mask3 = np.tile(mask[:, :, np.newaxis], [1, 1, PCA_th])

    # FISTA solver
    #
    solver = FISTA.FISTA(
        #
        f=lambda A: 1 / 2 * lin.norm((Y_PCA - dct.idct2d_bb(A))*mask3)**2,
        #
        df=lambda A: dct.dct2d_bb((dct.idct2d_bb(A) - Y_PCA)*mask3),
        #
        L=1,
        #
        g=lambda A: 0,
        #
        pg=lambda A: _proxg_refitting(A, Gamma),
        #
        shape=Y_PCA.shape,
        init=init,
        Nit=Nit,
        verbose=verbose)

    A_PCA, InfoOut_FISTA = solver.execute()

    #
    # Output managing.
    #
    InfoOut_CLS['E_CLS'] = InfoOut_CLS.pop('E')
    InfoOut_CLS['E_post_ls'] = InfoOut_FISTA['E']

    if PCA_transform:
        PCA_info = {
            'H': PCA_operator.H,
            'PCA_th': PCA_operator.PCA_th,
            'Ym': np.squeeze(PCA_operator.Ym[0, 0, :])
            }
        InfoOut_CLS['PCA_info'] = PCA_info

    X_PCA = dct.idct2d_bb(A_PCA)
    X_PCA = (X_PCA * Y_std) + Y_m
    Xhat = PCA_operator.inverse(X_PCA)

    return Xhat, InfoOut_CLS
