# -*- coding: utf-8 -*-

import unittest

import scipy.misc
import numpy.testing as npt
import numpy.linalg as lin

from context import pystem


class Test_sec2str(unittest.TestCase):

    def setUp(self):
        self.face = scipy.misc.face()
        self.face2D = self.face.sum(2)

    def test_dct2d(self):

        A = pystem.tools.dct.dct2d(self.face2D)
        X = pystem.tools.dct.idct2d(A)

        # Test direct and inverse is identity
        npt.assert_allclose(X, self.face2D)
        # Test that orthonormal transform
        npt.assert_allclose(lin.norm(A), lin.norm(self.face2D))

    def test_dct2d_bb(self):

        A = pystem.tools.dct.dct2d(self.face)
        X = pystem.tools.dct.idct2d(A)

        # Test direct and inverse is identity
        npt.assert_allclose(X, self.face, atol=1e-12)
        # Test that orthonormal transform
        npt.assert_allclose(lin.norm(A), lin.norm(self.face))
