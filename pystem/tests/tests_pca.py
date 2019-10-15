# -*- coding: utf-8 -*-

import unittest

import numpy as np
import numpy.random as npr
import numpy.testing as npt

from context import pystem


class Test_PCA(unittest.TestCase):

    def test_simple_pca(self):
        """
        """
        eigs = np.array([1e10, 5, 1e-6])  # , 1e-10, 1e-10, 1e-10])
        shape = (1000, 1000, eigs.size)
        Y = npr.randn(*shape) * np.broadcast_to(np.sqrt(eigs), shape)

        pca_obj = pystem.tools.PcaHandler(Y)
        Y_PCA, H, Ym, PCA_th = pca_obj.direct()

        npt.assert_allclose(eigs, pca_obj.InfoOut['d'])
