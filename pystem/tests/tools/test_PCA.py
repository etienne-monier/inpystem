# -*- coding: utf-8 -*-

import unittest

import numpy as np
import numpy.testing as npt
import numpy.random as npr

from ...tools import PCA


class Test_PCA(unittest.TestCase):

    def setUp(self):

        m, n, B = 10, 10, 3
        self.Y = np.arange(m*n*B).reshape((m, n, B))
        self.X = npr.randn(m, n, B)

    def test_EigenEstimate(self):
        pass

    def test_Dimension_Reduction(self):

        # Case with specific threshold.
        #
        th = 2
        S, InfoOut = PCA.Dimension_Reduction(self.Y, PCA_th=th)

        # Check th is correct
        self.assertEqual(th, InfoOut['PCA_th'])
        # Check H is ortho.
        H = InfoOut['H']
        npt.assert_allclose(np.dot(H.T, H), np.eye(th), atol=1e-10)
        # Check Ym
        m, n, B = self.Y.shape
        Ym = np.mean(self.Y.reshape((m*n, B)), axis=0)
        Ymr = np.tile(Ym[np.newaxis, np.newaxis, :], [m, n, 1])
        npt.assert_allclose(InfoOut['Ym'], Ymr)

        # Case PCA_th is 'max'
        #
        S, InfoOut = PCA.Dimension_Reduction(self.Y, PCA_th='max')
        self.assertEqual(InfoOut['PCA_th'], self.Y.shape[-1])

        # Case PCA_th is 'auto'
        #
        S, InfoOut = PCA.Dimension_Reduction(self.Y, PCA_th='auto')
        self.assertGreater(InfoOut['PCA_th'], 0)
        self.assertLessEqual(InfoOut['PCA_th'], self.Y.shape[-1])

        # Case less samples than dim.
        #
        N = 4
        Ytmp = np.moveaxis(self.Y, 0, -1)[:2, :2, :]
        S, InfoOut = PCA.Dimension_Reduction(Ytmp, PCA_th='max')
        self.assertEqual(InfoOut['PCA_th'], N-1)

        S, InfoOut = PCA.Dimension_Reduction(Ytmp, PCA_th='auto')
        self.assertEqual(InfoOut['PCA_th'], N-1)

    def test_PcaHandler(self):

        # Case where PCA_transform is False
        #
        op = PCA.PcaHandler(self.Y, PCA_transform=False)
        npt.assert_allclose(self.Y, op.Y_PCA)

        npt.assert_allclose(self.X, op.direct(self.X))
        npt.assert_allclose(self.X, op.inverse(self.X))

        # Case where PCA_transform is True
        #
        op = PCA.PcaHandler(self.Y, PCA_transform=True)
        npt.assert_allclose(self.Y, op.inverse(op.Y_PCA), atol=1e-10)
        npt.assert_allclose(op.direct(self.Y), op.Y_PCA, atol=1e-10)
