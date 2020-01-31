# -*- coding: utf-8 -*-

import unittest
import pathlib
import shutil

import numpy as np
import numpy.testing as npt

from ...restore import DL_ITKrMM as dl
from ...restore import DL_ITKrMM_matlab as dlm


class Test_DL_matlab(unittest.TestCase):
    """
    """
    def setUp(self):
        """Sets up tha data and shape.
        """

        # Create GT data.
        m, n, B = 10, 10, 3
        self.X = np.arange(m*n*B).reshape((m, n, B))

    def test_DL(self):
        """Starts wKSVD and ITKrMM without CLS nor save_it.

        It checks the output dico and data shapes are OK.
        """

        K, S, P = 5, 2, 7
        B = 3

        # ITKrMM
        Xhat, InfoOut = dlm.ITKrMM_matlab(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False,
            K=K, S=S, P=P)

        self.assertTupleEqual(InfoOut['dico'].shape, (K, P, P, B))
        self.assertTupleEqual(Xhat.shape, self.X.shape)

        # wKSVD
        Xhat, InfoOut = dlm.wKSVD_matlab(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False,
            K=K, S=S, P=P)

        self.assertTupleEqual(InfoOut['dico'].shape, (K, P, P, B))
        self.assertTupleEqual(Xhat.shape, self.X.shape)

    def test_DL_save_it(self):
        """Starts wKSVD and ITKrMM without CLS with save_it.

        It checks that the files are correctly saved.
        """

        K, S = 5, 2

        # Path where this is saved.
        p = pathlib.Path(dlm.__file__).parent / \
            'MatlabCodes' / 'ITKrMM' / 'save_it'

        # ITKrMM
        rec = dlm.Matlab_Dico_Learning_Executer(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False, K=K, S=S,
            save_it=True)
        Xhat, InfoOut = rec.execute()

        self.assertTrue((p / 'it_1.mat').exists())
        self.assertTrue((p / 'it_2.mat').exists())

        shutil.rmtree(p)

        # wKSVD
        rec = dlm.Matlab_Dico_Learning_Executer(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False, K=5, S=2,
            save_it=True)
        Xhat, InfoOut = rec.execute('wKSVD')

        self.assertTrue((p / 'it_1.mat').exists())
        self.assertTrue((p / 'it_2.mat').exists())

        shutil.rmtree(p)

    def test_DL_CLS(self):
        """Starts wKSVD and ITKrMM with CLS without save_it.

        It checks the estimated init shapes are OK and that lrc is returned
        without modification.
        """

        m, n, B = self.X.shape
        K, S, P = 5, 2, 5

        # ITKrMM
        rec = dlm.Matlab_Dico_Learning_Executer(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False, K=K, S=S,
            P=P, CLS_init={'Lambda': 0.1})

        # Check initialization shape is ok.
        init_lr, init = rec.get_CLS_init()
        self.assertEqual(init_lr.size, P**2*B)
        self.assertTupleEqual(init.shape, (P**2*B, K-1))

        Xhat, InfoOut = rec.execute()

        npt.assert_almost_equal(
            InfoOut['CLS_init'][0, :, :, :], InfoOut['dico'][0, :, :, :])

        # wKSVD
        rec = dlm.Matlab_Dico_Learning_Executer(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False, K=5, S=2,
            CLS_init={'Lambda': 0.1})
        Xhat, InfoOut = rec.execute('wKSVD')

        self.assertEqual(InfoOut['dico'].shape[2], K)
        self.assertTupleEqual(Xhat.shape, self.X.shape)


class Test_DL(unittest.TestCase):
    """
    """
    def setUp(self):
        """Sets up tha data and shape.
        """

        # Create GT data.
        m, n, B = 10, 10, 3
        self.X = np.arange(m*n*B).reshape((m, n, B))

    def test_DL(self):
        """Starts wKSVD and ITKrMM without CLS nor save_it.

        It checks the output dico and data shapes are OK.
        """

        K, S, P = 5, 2, 7
        B = 3

        # ITKrMM
        Xhat, InfoOut = dl.ITKrMM(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False,
            K=K, S=S, P=P)

        self.assertTupleEqual(InfoOut['dico'].shape, (K, P, P, B))
        self.assertTupleEqual(Xhat.shape, self.X.shape)

        # wKSVD
        Xhat, InfoOut = dl.wKSVD(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False,
            K=K, S=S, P=P)

        self.assertTupleEqual(InfoOut['dico'].shape, (K, P, P, B))
        self.assertTupleEqual(Xhat.shape, self.X.shape)

    def test_DL_CLS(self):
        """Starts wKSVD and ITKrMM with CLS without save_it.

        It checks the estimated init shapes are OK and that lrc is returned
        without modification.
        """

        m, n, B = self.X.shape
        K, S, P = 5, 2, 5

        # ITKrMM
        rec = dl.Dico_Learning_Executer(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False, K=K, S=S,
            P=P, CLS_init={'Lambda': 0.1})

        # Check initialization shape is ok.
        init_lr, init = rec.get_CLS_init()
        self.assertEqual(init_lr.size, P**2*B)
        self.assertTupleEqual(init.shape, (P**2*B, K-1))

        Xhat, InfoOut = rec.execute()

        npt.assert_almost_equal(
            InfoOut['CLS_init'][0, :, :, :], InfoOut['dico'][0, :, :, :])

        # wKSVD
        rec = dl.Dico_Learning_Executer(
            self.X, mask=None, Nit_lr=1, Nit=2, PCA_transform=False, K=5, S=2,
            CLS_init={'Lambda': 0.1})
        Xhat, InfoOut = rec.execute('wKSVD')

        self.assertEqual(InfoOut['dico'].shape[2], K)
        self.assertTupleEqual(Xhat.shape, self.X.shape)
