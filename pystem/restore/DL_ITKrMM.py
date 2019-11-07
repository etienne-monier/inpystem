# -*- coding: utf-8 -*-
"""This module implements the ITKrMM algorithm.
"""

import time
import pathlib
import logging
import functools
import multiprocessing as mp

import numpy as np
import numpy.random as rd
import numpy.linalg as lin

import scipy.sparse as sps

import sklearn
from sklearn.feature_extraction import image

from . import LS_CLS
from ..tools import PCA
from ..tools import matlab_interface as matlab
from ..tools import sec2str
from ..tools import metrics

_logger = logging.getLogger(__name__)


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


class Dico_Learning_Executer:
    """Class to define execute dictionary learning algorithms.

    The following class is a common code for most dictionary learning methods.
    It performs the following tasks:

    * reshapes the data in patch format,
    * performs low-rank component estimation,
    * launches the dictionary learning method,
    * reshape output data,
    * handle CLS initialization to speed-up computation.

    Attributes
    ----------
    Y: (m, n) or (m, n, l) numpy array
        The input data.
    Y_PCA: (m, n) or (m, n, PCA_th) numpy array
        The input data in PCA space.
        Its value is Y if Y is 2D.
    mask: (m, n) numpy array
        The acquisition mask.
    PatchSize: int
        The width (or height) of the patch.
        Default is 5.
    K: int
        The dictionary dimension.
        Default is 128.
    L: int
        The number of low rank components to learn.
        Default is 1.
    S: int
        The code sparsity level. Default is 20.
    Nit_lr: int
        The number of iterations for the low rank estimation.
        Default is 10.
    Nit: int
        The number of iterations. Default is 40.
    CLS_init: dico
        CLS initialization inofrmation. See Note for details.
        Default is None.
    xref: (m, n) or (m, n, l) numpy array
        Reference image to compute error evolution.
        Default is None for input Y data.
    verbose: bool
        The verbose parameter. Default is True.
    data: (PatchSize**2, N) or (PatchSize**2*l, N) numpy array
        The Y data in patch format. N is the number of patches.
    mdata: (PatchSize**2, N) or (PatchSize**2*l, N) numpy array
        The mask in patch format. N is the number of patches.
    init: (PatchSize**2, K+L) or (PatchSize**2*l, K+L) numpy array
        The initialization in patch format.
    PCA_operator: PcaHandler object
        The PCA operator.
    neam_std: 2-tuple
        Tuple of size 2 which contains the data mean and std.

    Note
    ----
        The algorithm can be initialized with CLS as soon as
        :code:`CLS_init` is not None.  In this case, :code:`CLS_init`
        should be a dictionary containing the required :code:`Lambda`
        key and other optional arguments required for CLS
        (:code:`PCA_transform`, :code:`PCA_th`, :code:`init`).
    """

    def __init__(self, Y, mask=None, PatchSize=5, K=128, L=1, S=20, Nit_lr=10,
                 Nit=40, CLS_init=None, xref=None, verbose=True,
                 PCA_transform=True, PCA_th='auto'):
        """
        Arguments
        ---------
        Y: (m, n) or (m, n, l) numpy array
            The input data.
        mask: optional, None, (m, n) numpy array
            The acquisition mask.
            Default is None for full sampling.
        PatchSize: int
            The width (or height) of the patch.
            Default is 5.
        K: int
            The dictionary dimension.
            Default is 128.
        L: int
            The number of low rank components to learn.
            Default is 1.
        S: int
            The code sparsity level. Default is 20. S should be
            less than the patch size.
        Nit_lr: int
            The number of iterations for the low rank estimation.
            Default is 10.
        Nit: int
            The number of iterations. Default is 40.
        CLS_init: dico
            CLS initialization inofrmation. See Note for details.
            Default is None.
        xref: (m, n) or (m, n, l) numpy array
            Reference image to compute error evolution.
            Default is None for input Y data.
        verbose: bool
            The verbose parameter. Default is True.
        PCA_transform: optional, bool
            Enables the PCA transformation if True, otherwise, no PCA
            transformation is processed.
            Default is True.
        PCA_th: optional, int, str
            The desired data dimension after dimension reduction.
            Possible values are 'auto' for automatic choice, 'max' for maximum
            value and an int value for user value.
            Default is 'auto'.

        Note
        ----
            The algorithm can be initialized with CLS as soon as
            :code:`CLS_init` is not None.  In this case, :code:`CLS_init`
            should be a dictionary containing the required :code:`Lambda`
            key and eventually the :code:`init` optional argument.
        """

        self.Y = Y

        if mask is None:
            mask = np.ones(Y.shape[:2])

        self.mask = mask
        self.PatchSize = PatchSize

        self.K = K
        self.L = L
        self.S = S

        self.Nit = Nit
        self.Nit_lr = Nit_lr

        self.CLS_init = CLS_init
        self.xref = xref

        self.verbose = verbose

        if CLS_init is not None and Y.ndim != 3:
            _logger.warning(
                'Dico learning will not be initialized with CLS as input data '
                'is not 3D. Random init used.')

        if (S > PatchSize**2 and Y.ndim == 2) or (
                S > PatchSize**2*Y.shape[-1] and Y.ndim == 3):
            raise ValueError('S input is smaller than the patch size.')

        # Perform PCA if Y is 3D
        if self.Y.ndim == 3:
            PCA_operator = PCA.PcaHandler(
                Y, mask, PCA_transform=PCA_transform, PCA_th=PCA_th,
                verbose=verbose)
            Y_PCA, PCA_th = PCA_operator.Y_PCA, PCA_operator.PCA_th

            self.PCA_operator = PCA_operator

            if CLS_init is not None and 'init' in CLS_init:
                self.CLS_init['init'] = PCA_operator.direct(
                    self.CLS_init['init'])
        else:
            Y_PCA = Y.copy()
            self.PCA_operator = None

        # Normalize and center
        Y_m, Y_std = Y_PCA.mean(), Y_PCA.std()
        Y_PCA = (Y_PCA - Y_m)/Y_std

        if CLS_init is not None and 'init' in CLS_init:
            self.CLS_init['init'] = (self.CLS_init['init'] - Y_m)/Y_std

        self.mean_std = (Y_m, Y_std)
        self.Y_PCA = Y_PCA

        # Prepare data
        obs_mask = mask if Y.ndim == 2 else np.tile(
            mask[:, :, np.newaxis], [1, 1, Y_PCA.shape[2]])

        # Observation
        self.data = forward_patch_transform(Y_PCA * obs_mask, self.PatchSize)

        # Mask
        self.mdata = forward_patch_transform(obs_mask, self.PatchSize)
        self.data *= self.mdata

        rd.seed(0)
        self.init = rd.randn(self.data.shape[0], self.K)

    def execute(self, method='ITKrMM'):
        """Executes dico learning restoration.

        Arguments
        ---------
        method: str
            The method to use, which can be 'ITKrMM' or 'wKSVD'.
            Default is 'ITKrMM'.

        Returns
        -------
        (m, n) or (m, n, l) numpy array
            Restored data.
        dict
            Aditional informations. See Notes.

        Note
        ----
             The output information keys are:
                - 'time': Execution time in seconds.
                - 'lrc': low rank component.
                - 'dico': Estimated dictionary.
                - 'E': Evolution of the error.
        """

        # Welcome message
        if self.verbose:
            print("-- {} reconstruction algorithm --".format(method))

        start = time.time()

        # Default initialization is random.
        # rd.seed(0)
        # self.init = rd.randn(self.data.shape[0], self.K)

        # Let us define the true number of dico atoms to search for init.
        if method == "wKSVD":
            K0 = self.K - self.L - 1  # As there's a DC component in wKSVD
        else:
            K0 = self.K - self.L  # For ITKrMM and others

        #
        # Learn lrc
        #
        if self.verbose:
            print('Learning low rank component...')

        if self.CLS_init is None or self.Y.ndim != 3:

            if self.L > 0:

                lrc = np.zeros((self.data.shape[0], self.L))

                for cnt in range(self.L):

                    lrc_init = self.init[:, cnt]

                    if cnt > 0:
                        lrc_init -= lrc[:, :cnt] @ lrc[:, :cnt].T @ lrc_init

                    lrc_init /= np.linalg.norm(lrc_init)

                    lrc[:, cnt] = rec_lratom(
                        self.data,
                        self.mdata,
                        lrc[:, :cnt] if cnt > 0 else None,
                        self.Nit_lr,
                        lrc_init)
            else:
                lrc = None

            dicoinit = self.init[:, self.L:self.L + K0]

        else:

            # Get initialization dictionary
            D, C, Xhat, InfoOut = CLS_init(
                self.Y_PCA,
                mask=self.mask,
                PatchSize=self.PatchSize,
                K=K0,
                S=self.S,
                PCA_transform=False,
                verbose=self.verbose,
                **self.CLS_init)

            # Get low rank component
            CLS_data = forward_patch_transform(Xhat, self.PatchSize)

            Uec, _, _ = np.linalg.svd(CLS_data)

            if self.L > 0:
                lrc = Uec[:, :self.L]
            else:
                lrc = None

            dicoinit = D.T
        #
        # Learn Dictionary
        #
        if self.verbose:
            print('Learning dictionary...'.format(method))

        # Remove lrc and ensures othogonality of input dico initialization.
        if self.L > 1:
            dicoinit -= lrc @ lrc.T @ dicoinit

        dicoinit = dicoinit @ np.diag(1 / lin.norm(dicoinit, axis=0))

        # Call reconstruction algo
        if method == 'ITKrMM':
            dico_hat, E, dt = itkrmm_core(
                self.data,
                masks=self.mdata,
                K=self.K,
                S=self.S,
                lrc=lrc,
                Nit=self.Nit,
                init=dicoinit,
                verbose=self.verbose,
                xref=self.xref)

            info = {'time': dt, 'E': E}

        elif method == 'wKSVD':
            dico_hat, info = wKSVD_core(
                self.data,
                masks=self.mdata,
                K=self.K,
                S=self.S,
                lrc=lrc,
                Nit=self.Nit,
                init=dicoinit,
                verbose=self.verbose,
                xref=self.xref,
                preserve_DC=True)

        else:
            raise ValueError(
                'Unknown method parameter for Dico_Learning_Executer object')

        #
        # Reconstruct data
        #
        # Recontruct data from dico and coeffs.
        coeffs = OMPm(dico_hat.T, self.data.T, self.S, self.mdata.T)
        outpatches = sps.csc_matrix.dot(dico_hat, (coeffs.T).tocsc())

        # Transform from patches to data.
        Xhat = inverse_patch_transform(outpatches, self.Y_PCA.shape)
        Xhat = Xhat * self.mean_std[1] + self.mean_std[0]

        if self.Y.ndim == 3:
            Xhat = self.PCA_operator.inverse(Xhat)

        # Reshape output dico
        p = self.PatchSize
        shape_dico = (self.K, p, p) if self.Y.ndim == 2 else (
            self.K, p, p, self.Y_PCA.shape[-1])

        dico = dico_hat.T.reshape(shape_dico)

        # Manage output info
        dt = time.time() - start
        InfoOut = {'dico': dico, 'E': info['E'], 'time': dt}

        if self.PCA_operator is not None:
            PCA_info = {
                'H': self.PCA_operator.H,
                'PCA_th': self.PCA_operator.PCA_th,
                'Ym': np.squeeze(self.PCA_operator.Ym[0, 0, :])
                }
            InfoOut['PCA_info'] = PCA_info

        if self.verbose:
            print(
                "Done in {}.\n---".format(sec2str.sec2str(dt)))

        return Xhat, InfoOut

    def execute_matlab(self, method='ITKrMM'):
        """Executes dico learning restoration.

        Arguments
        ---------
        method: str
            The method to use, which can be 'ITKrMM' or 'wKSVD'.
            Default is 'ITKrMM'.

        Returns
        -------
        (m, n) or (m, n, l) numpy array
            Restored data.
        dict
            Aditional informations. See Notes.

        Note
        ----
             The output information keys are:
                - 'time': Execution time in seconds.
                - 'lrc': low rank component.
                - 'dico': Estimated dictionary.
                - 'E': Evolution of the error.
        """
        # Welcome message
        if self.verbose:
            print("-- {}_matlab reconstruction algorithm --".format(method))

        # self.init = rd.randn(self.data.shape[0], self.K + self.L)

        #
        # Execute algorithm
        #
        if self.CLS_init is None:
            data = self.execute_no_CLS(method)
        else:
            data = self.execute_CLS(method)

        outPatches = data['outdata']

        #
        # Reconstruct data
        #

        # Transform from patches to data.
        Xhat = inverse_patch_transform(outPatches, self.Y_PCA.shape)

        Xhat = Xhat * self.mean_std[1] + self.mean_std[0]
        if self.Y.ndim == 3:
            Xhat = self.PCA_operator.inverse(Xhat)

        # Reshape output dico
        p = self.PatchSize
        shape_dico = (self.K, p, p) if self.Y.ndim == 2 else (
            self.K, p, p, self.Y_PCA.shape[-1])

        dico = data['dico'].T.reshape(shape_dico)

        # Manage output info
        dt = data['time']
        InfoOut = {'time': dt, 'dico': dico, 'E': data['E']}

        if self.PCA_operator is not None:
            PCA_info = {
                'H': self.PCA_operator.H,
                'PCA_th': self.PCA_operator.PCA_th,
                'Ym': np.squeeze(self.PCA_operator.Ym[0, 0, :])
                }
            InfoOut['PCA_info'] = PCA_info

        if self.verbose:
            print(
                "Done in {}.\n---".format(sec2str.sec2str(dt)))

        return Xhat, InfoOut

    def execute_no_CLS(self, method='ITKrMM'):
        """
        """
        # import ipdb; ipdb.set_trace()
        # Let us define the true number of dico atoms to search for init.
        if method == "wKSVD":
            K0 = self.K - self.L  # As there's a DC component in wKSVD
        else:
            K0 = self.K - self.L  # For ITKrMM and others

        # Arguments.
        Dico = {'corrpatches': self.data, 'maskpatches': self.mdata,
                'd': self.PatchSize * self.PatchSize,
                'K': K0,
                'L': self.L,
                'S': self.S,
                'X': self.data,
                'init': self.init,
                'maxit': self.Nit,
                'maxitLR': self.Nit_lr,
                'verbose': 1 if self.verbose else 0}

        # Executes program
        dirpath = pathlib.Path(__file__).parent / 'MatlabCodes' / 'ITKrMM'

        if method == 'ITKrMM':

            data = matlab.matlab_interface(
                dirpath / 'ITKrMM_for_python.m',
                Dico)

            lrc = data['lrc']
            data['dico'] = np.hstack((
                lrc if lrc.ndim == 2 else lrc[:, np.newaxis],
                data['dico']))

            return data

        elif method == 'wKSVD':

            return matlab.matlab_interface(
                dirpath / 'wKSVD_for_python.m',
                Dico)

        else:
            raise ValueError('Unknown method {}'.method)

    def execute_CLS(self, method='ITKrMM'):
        """
        """
        # Let us define the true number of dico atoms to search for init.
        if method == "wKSVD":
            K0 = self.K - self.L - 1  # As there's a DC component in wKSVD
        else:
            K0 = self.K - self.L  # For ITKrMM and others

        start = time.time()

        # Get initialization dictionary
        D, C, Xhat, InfoOut = CLS_init(
            self.Y_PCA,
            mask=self.mask,
            PatchSize=self.PatchSize,
            K=K0,
            S=self.S,
            PCA_transform=False,
            verbose=self.verbose,
            **self.CLS_init)

        # Get low rank component
        CLS_data = forward_patch_transform(Xhat, self.PatchSize)

        Uec, _, _ = np.linalg.svd(CLS_data)

        lrcomp = Uec[:, :self.L]
        dicoinit = D.T

        # Arguments.
        Dico = {'corrpatches': self.data, 'maskpatches': self.mdata,
                'd': self.PatchSize * self.PatchSize,
                'K': K0,
                'L': self.L,
                'S': self.S,
                'X': self.data,
                'lrcomp': lrcomp,
                'dicoinit': dicoinit,
                'maxit': self.Nit,
                'maxitLR': self.Nit_lr,
                'verbose': 1 if self.verbose else 0}

        # Executes program
        dirpath = pathlib.Path(__file__).parent / 'MatlabCodes' / 'ITKrMM'

        if method == 'ITKrMM':

            data = matlab.matlab_interface(
                dirpath / 'ITKrMM_CLS_init_for_python.m',
                Dico)

            lrc = data['lrc']
            data['dico'] = np.hstack((
                lrc if lrc.ndim == 2 else lrc[:, np.newaxis],
                data['dico']))

        elif method == 'wKSVD':
            data = matlab.matlab_interface(
                dirpath / 'wKSVD_init_CLS_for_python.m',
                Dico)

        else:
            raise ValueError('Unknown method {}'.method)

        data['time'] = time.time() - start

        return data


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


def ITKrMM(Y, mask=None, PatchSize=5, K=128, L=1, S=20, Nit_lr=10,
           Nit=40, CLS_init=None, xref=None, verbose=True,
           PCA_transform=True, PCA_th='auto'):
    """ITKrMM restoration algorithm.

    Arguments
    ---------
    Y: (m, n) or (m, n, l) numpy array
        The input data.
    mask: optional, None or (m, n) numpy array
        The acquisition mask.
        Default is None for full sampling.
    PatchSize: optional, int
        The width (or height) of the patch.
        Default is 5.
    K: optional, int
        The dictionary dimension.
        Default is 128.
    L: optional, int
        The number of low rank components to learn.
        Default is 1.
    S: optional, int
        The code sparsity level. Default is 20.
    Nit_lr: optional, int
        The number of iterations for the low rank estimation.
        Default is 10.
    Nit: optional, int
        The number of iterations. Default is 40.
    CLS_init: optional, dico
        CLS initialization inofrmation. See Notes for details.
        Default is None.
    xref: optional, (m, n) or (m, n, l) numpy array
        Reference image to compute error evolution.
        Default is None for input Y data.
    verbose: optional, bool
        The verbose parameter. Default is True.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is True.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction.
        Possible values are 'auto' for automatic choice, 'max' for maximum
        value and an int value for user value.
        Default is 'auto'.

    Returns
    -------
    (m, n) or (m, n, l) numpy array
        Restored data.
    dict
        Aditional informations. See Notes.

    Notes
    -----

        The algorithm can be initialized with CLS as soon as
        :code:`CLS_init` is not None.  In this case, :code:`CLS_init`
        should be a dictionary containing the required :code:`Lambda`
        key and eventually the :code:`init` optional argument.

        The output information keys are:

        * :code:`time`: Execution time in seconds.
        * :code:`lrc`: low rank component.
        * :code:`dico`: Estimated dictionary.
        * :code:`E`: Evolution of the error.
    """

    obj = Dico_Learning_Executer(
        Y, mask, PatchSize, K, L, S, Nit_lr,
        Nit, CLS_init, xref, verbose, PCA_transform, PCA_th)
    return obj.execute(method='ITKrMM')


def wKSVD(Y, mask=None, PatchSize=5, K=128, L=1, S=20, Nit_lr=10,
          Nit=40, CLS_init=None, xref=None, verbose=True,
          PCA_transform=True, PCA_th='auto'):
    """wKSVD restoration algorithm.

    Arguments
    ---------
    Y: (m, n) or (m, n, l) numpy array
        The input data.
    mask: optional, None or (m, n) numpy array
        The acquisition mask.
        Default is None for full sampling.
    PatchSize: optional, int
        The width (or height) of the patch.
        Default is 5.
    K: optional, int
        The dictionary dimension.
        Default is 128.
    L: optional, int
        The number of low rank components to learn.
        Default is 1.
    S: optional, int
        The code sparsity level. Default is 20.
    Nit_lr: optional, int
        The number of iterations for the low rank estimation.
        Default is 10.
    Nit: optional, int
        The number of iterations. Default is 40.
    CLS_init: optional, dico
        CLS initialization inofrmation. See Notes for details.
        Default is None.
    xref: optional, (m, n) or (m, n, l) numpy array
        Reference image to compute error evolution.
        Default is None for input Y data.
    verbose: optional, bool
        The verbose parameter. Default is True.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is True.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction.
        Possible values are 'auto' for automatic choice, 'max' for maximum
        value and an int value for user value.
        Default is 'auto'.

    Returns
    -------
    (m, n) or (m, n, l) numpy array
        Restored data.
    dict
        Aditional informations. See Notes.

    Notes
    -----

        The algorithm can be initialized with CLS as soon as
        :code:`CLS_init` is not None.  In this case, :code:`CLS_init`
        should be a dictionary containing the required :code:`Lambda`
        key and eventually the :code:`init` optional argument.

        The output information keys are:

        * :code:`time`: Execution time in seconds.
        * :code:`lrc`: low rank component.
        * :code:`dico`: Estimated dictionary.
        * :code:`E`: Evolution of the error.
    """

    obj = Dico_Learning_Executer(
        Y, mask, PatchSize, K, L, S, Nit_lr,
        Nit, CLS_init, xref, verbose, PCA_transform, PCA_th)
    return obj.execute(method='wKSVD')


def ITKrMM_matlab(Y, mask, PatchSize=5, K=128, L=1, S=20, Nit_lr=10,
                  Nit=40, CLS_init=None, xref=None, verbose=True,
                  PCA_transform=True, PCA_th='auto'):
    """ITKrMM restoration algorithm with matlab code.

    Arguments
    ---------
    Y: (m, n) or (m, n, l) numpy array
        The input data.
    mask: optional, None or (m, n) numpy array
        The acquisition mask.
        Default is None for full sampling.
    PatchSize: optional, int
        The width (or height) of the patch.
        Default is 5.
    K: optional, int
        The dictionary dimension.
        Default is 128.
    L: optional, int
        The number of low rank components to learn.
        Default is 1.
    S: optional, int
        The code sparsity level. Default is 20.
    Nit_lr: optional, int
        The number of iterations for the low rank estimation.
        Default is 10.
    Nit: optional, int
        The number of iterations. Default is 40.
    CLS_init: optional, dico
        CLS initialization inofrmation. See Notes for details.
        Default is None.
    xref: optional, (m, n) or (m, n, l) numpy array
        Reference image to compute error evolution.
        Default is None for input Y data.
    verbose: optional, bool
        The verbose parameter. Default is True.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is True.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction.
        Possible values are 'auto' for automatic choice, 'max' for maximum
        value and an int value for user value.
        Default is 'auto'.

    Returns
    -------
    (m, n) or (m, n, l) numpy array
        Restored data.
    dict
        Aditional informations. See Notes.

    Notes
    -----

        The algorithm can be initialized with CLS as soon as
        :code:`CLS_init` is not None.  In this case, :code:`CLS_init`
        should be a dictionary containing the required :code:`Lambda`
        key and eventually the :code:`init` optional argument.

        The output information keys are:

        * :code:`time`: Execution time in seconds.
        * :code:`lrc`: low rank component.
        * :code:`dico`: Estimated dictionary.
        * :code:`E`: Evolution of the error.
    """

    obj = Dico_Learning_Executer(
        Y, mask, PatchSize, K, L, S, Nit_lr,
        Nit, CLS_init, xref, verbose, PCA_transform, PCA_th)
    return obj.execute_matlab(method='ITKrMM')


def wKSVD_matlab(Y, mask, PatchSize=5, K=128, L=1, S=20, Nit_lr=10,
                 Nit=40, CLS_init=None, xref=None, verbose=True,
                 PCA_transform=True, PCA_th='auto'):
    """wKSVD restoration algorithm with Matlab code.

    Arguments
    ---------
    Y: (m, n) or (m, n, l) numpy array
        The input data.
    mask: optional, None or (m, n) numpy array
        The acquisition mask.
        Default is None for full sampling.
    PatchSize: optional, int
        The width (or height) of the patch.
        Default is 5.
    K: optional, int
        The dictionary dimension.
        Default is 128.
    L: optional, int
        The number of low rank components to learn.
        Default is 1.
    S: optional, int
        The code sparsity level. Default is 20.
    Nit_lr: optional, int
        The number of iterations for the low rank estimation.
        Default is 10.
    Nit: optional, int
        The number of iterations. Default is 40.
    CLS_init: optional, dico
        CLS initialization inofrmation. See Notes for details.
        Default is None.
    xref: optional, (m, n) or (m, n, l) numpy array
        Reference image to compute error evolution.
        Default is None for input Y data.
    verbose: optional, bool
        The verbose parameter. Default is True.
    PCA_transform: optional, bool
        Enables the PCA transformation if True, otherwise, no PCA
        transformation is processed.
        Default is True.
    PCA_th: optional, int, str
        The desired data dimension after dimension reduction.
        Possible values are 'auto' for automatic choice, 'max' for maximum
        value and an int value for user value.
        Default is 'auto'.

    Returns
    -------
    (m, n) or (m, n, l) numpy array
        Restored data.
    dict
        Aditional informations. See Notes.

    Notes
    -----

        The algorithm can be initialized with CLS as soon as
        :code:`CLS_init` is not None.  In this case, :code:`CLS_init`
        should be a dictionary containing the required :code:`Lambda`
        key and eventually the :code:`init` optional argument.

        The output information keys are:

        * :code:`time`: Execution time in seconds.
        * :code:`lrc`: low rank component.
        * :code:`dico`: Estimated dictionary.
        * :code:`E`: Evolution of the error.
    """

    obj = Dico_Learning_Executer(
        Y, mask, PatchSize, K, L, S, Nit_lr,
        Nit, CLS_init, xref, verbose, PCA_transform, PCA_th)
    return obj.execute_matlab(method='wKSVD')


def rec_lratom(data, masks=None, lrc=None, Nit=10, inatom=None, verbose=True):
    """Recover new low rank atom equivalent to itkrmm with K = S = 1.

    Arguments
    ---------
    data: (d, N) numpy array
        The (corrupted) training signals as its columns.
    masks: (d, N) numpy array
        Mask data as its columns.
        masks(.,.) in {0,1}.
        Default is masks = 1.
    lrc: (d, n) numpy array
        Orthobasis for already recovered low rank component.
        Default is None.
    Nit: int
        Number of iterations.
        Default is 10.
    inatom: (d, ) numpy array
        Initialisation that should be normalized.
        Default is None for random.
    verbose: bool
        If verbose is True, information is sent to the output.
        Default is True.

    Returns
    -------
    atom: (d, ) numpy array
        Estimated low rank component.
    """

    d, N = data.shape

    if masks is None:
        masks = np.ones((d, N))

    data = data*masks  # Safeguard

    # Create random initial point if needed or check input initialization is
    # normalized.
    if inatom is None:
        inatom = np.random.randn(d)

    inatom = inatom/np.linalg.norm(inatom)

    #
    if lrc is not None:

        # If lrc has 1 dimension, one should add a dimension to have correct
        # L.
        if lrc.ndim == 1:
            lrc = lrc[:, np.newaxis]
        L = lrc.shape[1]

        # Remove low rank component from initial atom and re-normalize.
        inatom = inatom - lrc @ lrc.T @ inatom
        inatom = inatom/np.linalg.norm(inatom)

        # Project data into orthogonal of lrc
        # start = time.time()
        for n in range(N):

            lrcMn = lrc * np.tile(masks[:, n][:, np.newaxis], [1, L])
            data[:, n] -= lrcMn @ np.linalg.pinv(lrcMn) @ data[:, n]

        # if verbose:
        #     print('Elapsed time: {}'.format(
        #         sec2str.sec2str(time.time()-start)))

    #
    # Start estimation

    atom_k = inatom

    for it in range(Nit):

        ip = atom_k.T.dot(data)
        maskw = np.sum(masks, 1)

        if lrc is None:
            atom_kp1 = data @ np.sign(ip).T

        else:
            atom_kp1 = np.zeros(atom_k.shape)

            for n in range(N):

                # The masked basis of the current low-rank space.
                lrcplus = np.concatenate(
                    (lrc, atom_k[:, np.newaxis]),
                    axis=1) * np.tile(masks[:, n][:, np.newaxis], [1, L+1])

                # The data is projected into the orthogonal space of lrcplus.
                resn = data[:, n] - \
                    lrcplus @ np.linalg.pinv(lrcplus) @ data[:, n]

                # The masked current estimated lrc.
                atom_k_mm = atom_k * masks[:, n]

                # Calculate incremented atom_kp1.
                atom_kp1 += \
                    np.sign(ip[n]) * resn + \
                    np.abs(ip[n])*atom_k_mm/np.sum(atom_k_mm**2)

        # Normalize  with mask score.
        if maskw.min() > 0:
            atom_kp1 /= maskw
        else:
            atom_kp1 /= (maskw + 1e-2)

        # Remove previous low rank components from current estimate.
        if lrc is not None:
            atom_kp1 -= lrc @ lrc.T @ atom_kp1

        # Re-normalize current estimation
        atom_kp1 /= np.linalg.norm(atom_kp1)

        # Update
        atom_k = atom_kp1

    return atom_k


def OMPm(D, X, S, Masks=None):
    r"""Masked OMP.

    This is a modified version of OMP to account for corruptions in the signal.

    Consider some input data :math:`\mathbf{X}` (whose shape is (N, P) where N
    is the number of signals) which are masked by :math:`\mathbf{M}`. Given an
    input dictionary :math:`\mathbf{D}` of shape (K, P), this algorithm returns
    the optimal sparse :math:`\hat{\mathbf{A}}` matrix such that:

    .. math::

        \gdef \A {\mathbf{A}}
        \gdef \M {\mathbf{M}}
        \gdef \X {\mathbf{X}}
        \gdef \D {\mathbf{D}}

        \begin{aligned}
        \hat{\A} &= \arg\min_\A \frac{1}{2}||\M\X - \M(\A\D)||_F^2\\
           &s.t. \max_k||\A_{k,:}||_{0} \leq S
        \end{aligned}

    A slightly different modification of Masked OMP is available in "Sparse
    and Redundant Representations: From Theory to Applications in Signal and
    Image Processing," the book written by M. Elad in 2010.

    Arguments
    ---------
    D: (K, P) numpy array
        The dictionary.
        Its rows MUST be normalized, i.e. their norm must be 1.
    X: (N, P) numpy array
        The masked signals to represent.
    S: int
        The max. number of coefficients for each signal.
    Masks: optional, (N, P) numpy array or None
        The sampling masks that should be 1 if sampled and 0 otherwise.
        Default is None for full sampling.

    Returns
    -------
    (N, K) sparse coo_matrix array
        sparse coefficient matrix.
    """

    # Get some dimensions
    N = X.shape[0]    # # of pixels in atoms
    P = X.shape[1]    # # of signals
    K = D.shape[0]    # # of atoms

    if Masks is None:
        Masks = np.ones((N, P))

    # Prepare the tables that will be used to create output sparse matrix.
    iTab = np.zeros(N*S)
    jTab = np.zeros(N*S)
    dataTab = np.zeros(N*S)
    Ncomp = 0  # Count the number of nnz elements for output.

    for k in range(N):
        # Local mask and signal # k
        x = X[k, :]
        m = Masks[k, :]
        xm = x*m  # Masked data

        # Masked atoms
        Dm = D * np.tile(m[np.newaxis, :], [K, 1])

        # Normalization of available masked atoms
        scale = np.linalg.norm(Dm, axis=1)
        nz = np.flatnonzero(scale > 1e-3 / np.sqrt(N))
        scale[nz] = 1/scale[nz]

        # Initialize residuals
        residual = xm

        # Initialize the sequence of atom indexes
        indx = np.zeros(S, dtype=int)

        for j in range(S):

            # Projection of the residual into dico
            proj = scale * (Dm @ residual)

            # Search max scalar product
            indx[j] = np.argmax(np.abs(proj))

            # Update residual
            a = np.linalg.pinv(Dm[indx[:j+1], :].T) @ xm
            residual = xm - Dm[indx[:j+1], :].T @ a

            # In case of small residual, break
            if np.linalg.norm(residual)**2 < 1e-6:
                break

        iTab[Ncomp:Ncomp+j+1] = k * np.ones(j+1)
        jTab[Ncomp:Ncomp+j+1] = indx[:j+1]
        dataTab[Ncomp:Ncomp+j+1] = a
        Ncomp += j+1

    # Build sparse output as scipy.sparse.coo_matrix
    return sps.coo_matrix((dataTab, (iTab, jTab)), shape=(N, K))


def _itkrmm_multi(n, lrc, data, masks, L):
    """
    """
    lrcMn = lrc * np.tile(masks[:, n][:, np.newaxis], [1, L])
    return lrcMn @ np.linalg.pinv(lrcMn) @ data[:, n]


def itkrmm_core(
        data, masks=None, K=None, S=1, lrc=None, Nit=50, init=None,
        verbose=True, xref=None):
    """Iterative Thresholding and K residual Means masked.

    Arguments
    ---------
    data: (d, N) numpy array
        The (corrupted) training signals as its columns.
    masks: optional, None, (d, N) numpy array
        The masks as its columns.
        masks(.,.) in {0,1}.
        Default is None for full sampling.
    K: optional, None or int
        Dictionary size.
        Default is None for d.
    S: optional, int
        Desired or estimated sparsity level of the signals.
        Default is 1.
    lrc: optional, None or (d, L) numpy array
        Orthobasis for low rank component. Default is None.
    Nit: optional, int
        Number of iterations.
        Default is 50.
    init: optional, None or (d, K-L) numpy array
        Initialisation, unit norm column matrix.
        Here, L is the number of low rank components.
        Default is None for random.
    verbose: optional, optional, bool
        The verbose parameter.
        Default is True.
    xref: optional, None or (d, N) numpy array
        True data or reference data to check convergence.
        Default is data.

    Returns
    -------
    (d, K) numpy array
        Estimated dictionary
    float
        Execution time in seconds.


    (Nit, ) numpy array
        Evolution of NMSE between current data and xref.
    """

    # d is patch size, N is # of patches.
    d, N = data.shape

    if masks is None:
        masks = np.ones(data.shape)

    data = data*masks  # safeguard

    if K is None:
        K = data.shape[0]

    if lrc is not None:
        L = 1 if lrc.ndim == 1 else lrc.shape[1]
        K = K - L

    if N < K-1:
        _logger.warning(
            'Less training signals than atoms: trivial solution is data.')
        return data, None

    if init is not None and not np.array_equal(init.shape, np.array([d, K])):
        _logger.warning(
            'Initialisation does not match dictionary shape. '
            'Random initialisation used.')
        init = None

    if init is None:
        init = np.random.randn(d, K)
    # Normalization of the columns
    init = init.dot(np.diag(1/lin.norm(init, axis=0)))

    # if xref is None:
    #     xref = data

    # Start algorithm --------------
    #
    start_0 = time.time()

    if lrc is not None:

        if lrc.ndim == 1:
            lrc = lrc[:, np.newaxis]

        L = lrc.shape[1]

        # Remove lrc from init and normalize columns.
        init = init - lrc @ lrc.T @ init
        init = init.dot(np.diag(1/lin.norm(init, axis=0)))

        # Remove lrc from data
        # start = time.time()
        pool = mp.Pool(processes=mp.cpu_count())

        f = functools.partial(
            _itkrmm_multi, lrc=lrc, data=data, masks=masks, L=L)

        res = pool.map(f, range(N))
        data -= np.asarray(res).T

        # if verbose:
        #     print('elapsed time: {}'.format(
        #         sec2str.sec2str(time.time()-start)))

    # Learn dictionary --------------
    #
    dico_k = init
    E = np.zeros(Nit)
    time_step = 0

    for it in range(Nit):

        # Print information
        if verbose:
            if it == 0:
                print('Iteration #{} over {}.'.format(it, Nit))
            else:
                print(
                    'Iteration #{} over {}'.format(it, Nit),
                    ' (estimated remaining time: ',
                    '{}). E: {:.2f}.'.format(
                        sec2str.sec2str(
                            time_step*(Nit-it+1))), E[it-1])

        start = time.time()

        # Learn dictionary
        #
        # Init.
        dico_kp1, maskw = np.zeros((d, K)), np.zeros((d, K))

        for n in range(N):  # N

            # Get support of mask for patch #n.
            supp = np.flatnonzero(masks[:, n])
            if supp.size == 0:
                continue

            #
            # Thresholding

            # Project data into dico to get code.
            # The dictionary is normalized with the norm of masked dico.
            dico_k_norm = lin.norm(dico_k[supp, :], axis=0)
            ipn = dico_k.T @ data[:, n] / dico_k_norm

            # Find support Int.
            absipn = np.abs(ipn)
            signipn = np.sign(ipn)
            In = np.argsort(absipn, axis=0)[::-1]
            Int = In[:S]

            #
            # Dico learning

            # Renormalised corrupted dico on support.
            masks_t = np.tile(masks[:, n], [S, 1]).T
            dInm = (dico_k[:, Int] * masks_t) @ np.diag(
                1/dico_k_norm[Int])

            # Construct residuals
            if lrc is not None:
                dico_LMn = lrc * np.tile(masks[:, n], [L, 1]).T
                dILnm = np.concatenate((dico_LMn, dInm), axis=1)
                resn = np.real(
                    data[:, n] - np.linalg.pinv(dILnm).T @
                    np.concatenate((np.zeros(L), ipn[Int]), axis=0)
                    )

            else:
                resn = np.real(data[:, n] - np.linalg.pinv(dInm).T @ ipn[Int])

            # Update new dictionary and maskweight
            dico_kp1[:, Int] += \
                resn[:, np.newaxis] @ signipn[np.newaxis, Int] +\
                dInm @ np.diag(absipn[Int])

            maskw[:, Int] += np.tile(masks[:, n], [S, 1]).T

        if maskw.min() > 0:
            dico_kp1 = N * dico_kp1 / maskw
        else:
            dico_kp1 = N * dico_kp1 / (maskw + 1e-3)

        if lrc is not None:
            dico_kp1 = dico_kp1 - lrc @ lrc.T @ dico_kp1

        # Compute the dico norm.
        scale = lin.norm(dico_kp1, axis=0)

        # Redraw atoms that are not used
        Iz = np.flatnonzero(scale**2 < 1e-5)
        dico_kp1[:, Iz] = rd.randn(d, Iz.size)
        scale = lin.norm(dico_kp1, axis=0)

        # Normalize
        dico_kp1 = dico_kp1 @ np.diag(1/scale)

        # Update
        dico_k = dico_kp1

        # Compute error
        if xref is not None:

            # Estimate coeffs with OMPm
            lrcdico = np.concatenate((lrc, dico_k), axis=1)
            coeffs = OMPm(lrcdico.T, data.T, S, masks.T)

            # Reconstruct data
            data_hat_patches = sps.csc_matrix.dot(lrcdico, (coeffs.T).tocsc())

            # Put it back into image shape
            m, n, B = xref.shape
            Xhat = inverse_patch_transform(data_hat_patches, xref.shape)

            # Compute error
            E[it] = metrics.SNR(xhat=Xhat, xref=xref)

        time_step = time.time() - start

    return np.concatenate((lrc, dico_k), axis=1), E, time.time() - start_0


def improve_atom(data, masks, dico, coeffs, j):
    """This function performs dictionary update for atom #j.

    In case the j'th atom is not used, a new atom is chosen among the data
    and the third output is set to True (False otherwise).

    Arguments
    ---------
    data: (d, N) numpy array
        The (corrupted) training signals as its columns.
    masks: (d, N) numpy array
        The masks as its columns.
    dico: (d, K) numpy array
        Initialisation, unit norm column matrix.
    coeffs: (K, N) numpy array
        The sparse codding.
    j: int
        The atom indice to update.

    Returns
    -------
    (d, ) numpy array
        The updated atom.
    (K, N) numpy array
        The updated atoms
    redrawn: int
        1 if a new atom has been generated, 0 therwise.
    """

    # All data indices i that uses the j'th dictionary element,
    # i.e. s.t. coeffs[j, i] != 0.
    nnz = coeffs[j, :].nonzero()[1]  # np.flatnonzero(coeffs[j, :])

    if nnz.size == 0:
        # No data uses the j'th atom.
        # In this case, this atom should be replaced.
        #
        # To replace this atom, the data which is has the greatest
        # reconstruction error should be chosen.

        error = data - dico @ coeffs
        error_norm = np.sum(error**2, axis=0)
        pos = np.argmax(error_norm)

        best_atom = data[:, pos]  # other possibility: error[:,pos]

        # Normalization
        best_atom = best_atom / np.linalg.norm(best_atom)
        if best_atom[0] != 0:
            best_atom *= np.sign(best_atom[0])

        M = coeffs.shape[1]
        coeffs[j, :] = sps.coo_matrix((1, M), dtype=np.float64)
        redrawn = 1

    else:

        redrawn = 0

        tmp_coeffs = coeffs[:, nnz]

        # The coefficients of the element we now improve are not relevant.
        tmp_coeffs[j, :] = 0

        # Vector of errors that we want to minimize with the new element.
        errors = data[:, nnz] - dico*tmp_coeffs

        #
        # wKSVD update:
        #    min || beta.*(errors - atom*coeff) ||_F^2 for beta = mask
        #
        Nit = 10  # The paper suggests 10-20 but 10 is fine and faster.

        best_atom = np.zeros((dico.shape[0], 1))
        coeff_new = np.zeros((1, nnz.size))

        for i in range(Nit):

            NewF = \
                masks[:, nnz]*errors + \
                (np.ones((masks.shape[0], nnz.size)) -
                    masks[:, nnz])*(
                        best_atom.dot(coeff_new))

            if nnz.size > 1:
                [best_atom, s, beta] = sps.linalg.svds(
                    sps.coo_matrix(NewF), 1)
            else:
                s = np.linalg.norm(NewF)
                beta = np.array([[1.0]])
                best_atom = NewF / s

            coeff_new = s * beta

        # The output atom is squeezed and if the first element is
        # nonzero, that's put to positive.
        best_atom = np.squeeze(best_atom)
        if best_atom[0] != 0:
            sign_atom = np.sign(best_atom[0])
            best_atom *= sign_atom
            coeff_new *= sign_atom

        coeffs[j, :] = sps.coo_matrix(
            (np.squeeze(coeff_new, axis=0),
                (np.zeros(nnz.size), nnz)),
            shape=(1, coeffs.shape[1]))

    return np.squeeze(best_atom), coeffs, redrawn


def dico_cleanup(data, dico, coeffs):
    """This function replaces all atoms:

    - which have a twin which is too close
    - which is not enough used to represent the data

    with data which have maximal representation error.

    Arguments
    ---------
    data: (d, N) numpy array
        The (corrupted) training signals as its columns.
    dico: (d, K) numpy array
        Initialisation, unit norm column matrix.
    coeffs: (K, N) numpy array
        The sparse codding.

    Returns
    -------
    (d, N) numpy array
        The dictionary after cleanup.
    """

    T2 = 0.99
    T1 = 3

    # # of atoms in dico.
    K = dico.shape[1]

    # The approx. error for all data. That's a (N, ) array.
    Er = sum((data - dico @ coeffs)**2, 0)

    # G[i, j] = | <dico_i, dico_j> if i != j
    #           | 0 otherwise.
    G = dico.T @ dico
    G -= np.diag(np.diag(G))

    for k in range(K):

        # If :
        #   - the atom #k has a twin which is too close
        #   - less than T1 data use the atom #k for their representation code
        #
        # Then the atom #k is replaced by the data which have maximal
        # representation error.
        if np.max(G[k, :]) > T2 or np.sum(np.abs(coeffs[k, :]) > 1e-7) <= T1:

            # Get the index of the data which have the maximal
            # representation error.
            pos = np.argmax(Er)

            # Its error is put to 0 not to be choosen in the future.
            Er[pos] = 0

            # Update new atom
            dico[:, k] = data[:, pos]/np.linalg.norm(data[:, pos])

            # Update G.
            G = dico.T @ dico
            G -= np.diag(np.diag(G))

    return dico


def wKSVD_core(
        data, masks=None, K=None, S=1, lrc=None, Nit=50, init=None,
        verbose=True, xref=None, preserve_DC=True):

    # Data and masks
    #
    # d is patch size, N is # of patches.
    d, N = data.shape

    if masks is None:
        masks = np.ones(data.shape)

    data = data*masks  # safeguard

    # Dico init
    #

    # Get dico dimension K.
    if K is None and init is None:
        K = data.shape[0]
    elif K is None and init is not None:
        K = init.shape[1]

    # Get lrc dimension L.
    if lrc is not None:
        if lrc.ndim == 1:
            L = 1
        else:
            L = lrc.shape[1]
    else:
        L = 0

    # Gets the dim of DCatom
    K_DC = 1 if preserve_DC else 0

    if N < K-1:
        _logger.warning(
            'Less training signals than atoms: trivial solution is data.')
        return data, None

    if init is not None and not np.array_equal(
            init.shape, np.array([d, K - L - K_DC])):
        _logger.warning(
            'Initialisation does not match dictionary shape. '
            'Random initialisation used.')
        init = None

    if init is None:
        init = np.random.randn(d, K - L - K_DC)

    # Initialize dico.
    if lrc is not None:
        dico = np.hstack((lrc, init))
    else:
        dico = init

    # dico is normalized.
    dico /= np.tile(np.linalg.norm(dico, axis=0), [d, 1])

    # In case preserve_DC is True, compute it and remove it from
    # initialization.
    if preserve_DC:
        DC_atom = np.ones(d)/np.sqrt(d)
        DC_coeffs = np.dot(DC_atom, dico)
        dico -= DC_atom[:, np.newaxis] @ DC_coeffs[np.newaxis, :]

    # dico is normalized (again as DC component has been removed)
    # and first line is set to be positive.
    dico /= np.tile(
        np.linalg.norm(dico, axis=0) * np.sign(dico[0, :]),
        [d, 1])

    if xref is None:
        xref = data

    #
    # The K-SVD algorithm starts here.
    #

    for it in range(Nit):

        start = time.time()

        if verbose:
            print('Iteration #{} over {}.'.format(it, Nit))

        # Sparse approximation using OMPm with fixed sparsity level S.
        #
        if preserve_DC:
            over_dico = np.hstack((DC_atom[:, np.newaxis], dico))
        else:
            over_dico = dico

        coeffs = OMPm(over_dico.T, data.T, S, masks.T)
        coeffs = coeffs.T.tocsr()

        # Dictionary update
        #
        redrawn_cnt = 0

        # Choose a random permutation to scan the dico.
        rPerm = np.random.permutation(dico.shape[1])

        for j in rPerm:  # range(dico.shape[1]):

            # Update the j'th atom
            best_atom, coeffs, redrawn = improve_atom(
                data,
                masks,
                over_dico,
                coeffs,
                j+1)

            # If DC atom is preserved, remove it from best_atom.
            if preserve_DC:

                DC_coeff = np.dot(DC_atom, best_atom)
                best_atom -= DC_coeff * DC_atom
                best_atom /= np.linalg.norm(best_atom)

            # Update atom in dico.
            dico[:, j] = best_atom

            # Increment the counter for redrawn atoms.
            redrawn_cnt = redrawn_cnt + redrawn

        dt = time.time() - start

        # This is to remove atoms :
        #   - which have a twin which is too close
        #   - which is not enough used to represent the data.
        dico = dico_cleanup(data, dico, coeffs[1:, :])

    dico_hat = np.hstack(
        (DC_atom[:, np.newaxis], dico)) if preserve_DC else dico

    return dico_hat, {'time': dt, 'redrawn_cnt': redrawn_cnt, 'E': 0}
