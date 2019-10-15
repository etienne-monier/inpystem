# -*- coding: utf-8 -*-

import unittest

import numpy as np
import numpy.testing as npt

from context import pystem


class Test_FISTA(unittest.TestCase):

    def test_fista_1(self):
        """First simple test for FISTA.

        Optimization problem is arg min_x {x^2} with x scalar.
        """
        #
        # With Nit
        Nit = 1000
        fista_obj = pystem.tools.FISTA(
            f=lambda x: x**2/2,
            df=lambda x: x,
            L=1,
            g=lambda x: 0,
            pg=lambda x: x,
            shape=(1,),
            Nit=Nit)

        xhat, InfoOut = fista_obj.execute()

        # Check StopTest
        self.assertTrue(fista_obj.StopTest(0))
        self.assertTrue(fista_obj.StopTest(10))
        self.assertTrue(fista_obj.StopTest(Nit-1))
        self.assertFalse(fista_obj.StopTest(Nit))

        # Check output
        self.assertEqual(InfoOut['E'].size, Nit)
        self.assertGreater(InfoOut['time'], 0)
        npt.assert_allclose(xhat, 0)

        #
        # Without Nit
        fista_obj = pystem.tools.FISTA(
            f=lambda x: x**2/2,
            df=lambda x: x,
            L=1,
            g=lambda x: 0,
            pg=lambda x: x,
            shape=(1,),)

        xhat, InfoOut = fista_obj.execute()

        # Check StopTest
        self.assertTrue(fista_obj.StopTest(0))
        self.assertTrue(fista_obj.StopTest(1))

        num_it = InfoOut['E'].size
        self.assertTrue(fista_obj.StopTest(num_it-1))
        self.assertFalse(fista_obj.StopTest(num_it))
        self.assertGreater(fista_obj.StopCritera(num_it-1),
                           fista_obj.lim)
        self.assertLess(fista_obj.StopCritera(num_it),
                        fista_obj.lim)

        # Check output
        self.assertGreater(InfoOut['time'], 0)
        self.assertLess(xhat, 5e-6)

    def test_fista_2(self):
        """Second simple test for FISTA.

        Optimization problem is arg min_x {x^2} with x a 2-numpy array.
        """
        #
        # With Nit
        Nit = 1000
        fista_obj = pystem.tools.FISTA(
            f=lambda x: (x[0]**2 + x[1]**2)/2,
            df=lambda x: x,
            L=1,
            g=lambda x: 0,
            pg=lambda x: x,
            shape=(2,),
            Nit=Nit)

        xhat, InfoOut = fista_obj.execute()

        # Check output
        npt.assert_allclose(xhat, np.array([0, 0]))
