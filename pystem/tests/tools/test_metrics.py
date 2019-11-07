# -*- coding: utf-8 -*-

import unittest

import numpy as np
import numpy.testing as npt

from ...tools import metrics


class Test_metrics(unittest.TestCase):

    def test_SNR(self):
        """
        """
        npt.assert_equal(metrics.SNR(xhat=0, xref=1), 0)

    def test_NMSE(self):
        """
        """
        npt.assert_equal(metrics.NMSE(xhat=0, xref=1), 1)
        npt.assert_equal(metrics.NMSE(xhat=1, xref=1), 0)

    def test_aSAD(self):
        """
        """
        # 1D case
        a = np.arange(5)
        b = np.array([10, 5, 8, 9, 1])/271
        expected = np.arccos(
            np.dot(a.T, b) /
            (np.linalg.norm(a)*np.linalg.norm(b))
            )

        npt.assert_allclose(
            metrics.aSAD(a, a),
            0
            )

        npt.assert_allclose(
            metrics.aSAD(a, b),
            expected
            )

        # 2D case
        A = np.tile(a, [2, 1])
        AB = np.vstack((a, b))

        npt.assert_allclose(
            metrics.aSAD(A, A),
            0
            )

        npt.assert_allclose(
            metrics.aSAD(AB, AB),
            0
            )

        npt.assert_allclose(
            metrics.aSAD(A, AB),
            expected/2
            )
