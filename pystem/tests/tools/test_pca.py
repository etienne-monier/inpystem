# -*- coding: utf-8 -*-

import unittest

import numpy as np
import numpy.random as npr
import numpy.testing as npt

from ...tools import PCA


class Test_PCA(unittest.TestCase):

    def test_EigenEstimate(self):
        """
        """
        Ns, M = 10, 100
        lin = np.ones(M)
        lout, sigma, D = PCA.EigenEstimate(lin, Ns)

        # Output are str. positive
        npt.assert_array_less(np.zeros(M), lout)

        # sigma is strct. positive
        self.assertGreater(sigma, 0)

        # D is between 1 and M-1
        self.assertGreater(D, 0)
        self.assertLess(D, M)

    def test_simple_pca(self):
        """
        """
        eigs = np.array([1e10, 5, 1e-6])  # , 1e-10, 1e-10, 1e-10])
        shape = (1000, 1000, eigs.size)
        Y = npr.randn(*shape) * np.broadcast_to(np.sqrt(eigs), shape)

        pca_obj = PCA.PcaHandler(Y)
        Y_PCA = pca_obj.direct()

        npt.assert_allclose(eigs, pca_obj.InfoOut['d'])
