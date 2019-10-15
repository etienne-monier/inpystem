# -*- coding: utf-8 -*-
"""This module gathers regularized least square restoration methods
adapted to 2D data.

The only method it implements for the moment is the **L1-LS** algorithm.
"""

import time

import numpy as np
import numpy.linalg as lin

from ..tools import FISTA
from ..tools import dct
from ..tools import sec2str


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


def L1_LS(Y, Lambda, mask=None, init=None, Nit=None, verbose=True):
    r"""L1-LS algorithm.

    The L1-LS algorithm denoises or reconstructs an image possibly spatially
    sub-sampled in the case of spatially sparse content in the DCT basis.
    It is well adapted to periodic data.

    This algorithms solves the folowing regularization problem:

    .. math::

       \gdef \x {\mathbf{x}}
       \gdef \y {\mathbf{y}}
       \hat{\x} = \mathrm{arg}\min_{ \x\in\mathbb{R}^{m \times n} }
           \frac{1}{2} ||(\x-\y)\cdot \Phi||_F^2 +
           \lambda ||\x\Psi||_1

    where :math:`\mathbf{y}` are the corrupted data,  :math:`\Phi` is a
    subsampling operator and :math:`\Psi` is a 2D DCT operator.

    Caution
    -------
    It is strongly recomended to remove the mean before reconstruction.
    Otherwise, this value could be lost automaticaly by the algorithm in
    case of powerful high frequencies.

    In the same way, normalizing the data is a good practice to have
    the parameter be low sensitive to data.

    **These two operations are implemented in this function.**

    Arguments
    ---------
    Y (m, n) numpy array
        An image which mean has been removed.
    Lambda: float
        Regularization parameter.
    mask: optional, None, (m, n) numpy array
        A sampling mask which is True if the pixel is sampled.
        Default is None for full sampling.
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
    (m, n) numpy array
        The reconstructed/denoised image.
    dict
        A dictionary containing some extra info

    Note
    ----
    Infos in output dictionary:

    * :code:`E`: In the case of partial reconstruction, the cost
      function evolution over iterations.
    * :code:`Gamma`: The array of kept coefficients (order is
      Fortran-style).
    * :code:`nnz_ratio`: the ratio Gamma.size/(m*n).
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
        print("-- L1-LS reconstruction algorithm --")

    # Center and normalize the data
    data_m, data_std = Y.mean(), Y.std()
    init_m, init_std = init.mean(), init.std()

    Y = (Y - data_m)/data_std
    init = (init - init_m)/init_std

    #
    # Separates denoising vs. inpainting
    #
    m, n = Y.shape
    N = mask.sum()
    P = m*n

    Y_d = dct.dct2d(Y)
    init_d = dct.dct2d(init)

    start = time.time()

    if (N == P):

        # Denoising
        #
        # In this case, the procedure consists in simply applying the g prox
        # operator to Y
        #

        A = _soft_thresholding(Y_d, Lambda)
        localInfo = {}

    else:

        # Inpainting
        #

        L = 1

        # FISTA solver
        #
        solver = FISTA.FISTA(
            #
            f=lambda A: 1 / 2 * lin.norm((Y - dct.idct2d(A))*mask)**2,
            #
            df=lambda A: dct.dct2d((dct.idct2d(A) - Y)*mask),
            #
            L=L,
            #
            g=lambda A: Lambda * np.sum(np.abs(A)),
            #
            pg=lambda A: _soft_thresholding(A, Lambda / L),
            #
            shape=Y_d.shape,
            init=init_d,
            Nit=Nit,
            verbose=verbose)

        A, InfoOut_FISTA = solver.execute()

        localInfo = {'E': InfoOut_FISTA['E']}

    #
    # Output managing.
    #

    X = dct.idct2d(A)
    Gamma = np.flatnonzero(A)
    nnz_ratio = Gamma.size / A.size

    # Add previous mean and std
    X = X * data_std + data_m

    dt = time.time() - start
    commonInfo = {'Gamma': Gamma,
                  'nnz_ratio': nnz_ratio,
                  'time': dt}

    InfoOut = {**localInfo, **commonInfo}

    if (verbose):
        print(
            """Done in {}.
Final ratio of nonzero coefficients is {:.3f} ({} nonzero coefficients over {})
---
""".format(sec2str.sec2str(dt), nnz_ratio, Gamma.size, P))

    return X, InfoOut
