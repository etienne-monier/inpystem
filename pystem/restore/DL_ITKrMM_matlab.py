# -*- coding: utf-8 -*-
"""This module implements the ITKrMM algorithm.
"""

import time
import pathlib
import logging

import numpy as np
import numpy.random as rd

from ..tools import PCA
from ..tools import matlab_interface as matlab
from ..tools import sec2str
from .DL_ITKrMM import forward_patch_transform, inverse_patch_transform, \
    CLS_init

_logger = logging.getLogger(__name__)


class Matlab_Dico_Learning_Executer:
    """Class to define and execute dictionary learning algorithms with
    matlab interface.

    The following class is a common code for most dictionary learning
    methods.
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
                 Nit=40, init=None, CLS_init=None, xref=None, verbose=True,
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
        init: (PatchSize**2, K+L) or (PatchSize**2*l, K+L) numpy array
            Initialization dictionary.
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

        if init is None:
            self.init = rd.randn(self.data.shape[0], self.K)
        else:
            self.init = init

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


def ITKrMM_matlab(Y, mask, PatchSize=5, K=128, L=1, S=20, Nit_lr=10,
                  Nit=40, init=None, CLS_init=None, xref=None, verbose=True,
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
    init: (PatchSize**2, K+L) or (PatchSize**2*l, K+L) numpy array
        Initialization dictionary.
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

    obj = Matlab_Dico_Learning_Executer(
        Y, mask, PatchSize, K, L, S, Nit_lr,
        Nit, init, CLS_init, xref, verbose, PCA_transform, PCA_th)
    return obj.execute(method='ITKrMM')


def wKSVD_matlab(Y, mask, PatchSize=5, K=128, L=1, S=20, Nit_lr=10,
                 Nit=40, init=None, CLS_init=None, xref=None, verbose=True,
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
    init: (PatchSize**2, K+L) or (PatchSize**2*l, K+L) numpy array
        Initialization dictionary.
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

    obj = Matlab_Dico_Learning_Executer(
        Y, mask, PatchSize, K, L, S, Nit_lr,
        Nit, init, CLS_init, xref, verbose, PCA_transform, PCA_th)
    return obj.execute(method='wKSVD')
