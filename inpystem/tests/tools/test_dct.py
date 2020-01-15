# -*- coding: utf-8 -*-

import unittest

import scipy.misc
import numpy.testing as npt
import numpy.linalg as lin

from ...tools import dct


class Test_dct(unittest.TestCase):

    def setUp(self):
        self.face = scipy.misc.face()
        self.face2D = self.face.sum(2)

    def test_dct2d(self):

        A = dct.dct2d(self.face2D)
        X = dct.idct2d(A)

        # Test direct followed by inverse is identity.
        npt.assert_allclose(X, self.face2D)
        # Test that the transformation is orthonormal.
        npt.assert_allclose(lin.norm(A), lin.norm(self.face2D))

    def test_dct2d_bb(self):

        A = dct.dct2d(self.face)
        X = dct.idct2d(A)

        # Test direct followed by inverse is identity.
        npt.assert_allclose(X, self.face, atol=1e-12)
        # Test that the transformation is orthonormal.
        npt.assert_allclose(lin.norm(A), lin.norm(self.face))
