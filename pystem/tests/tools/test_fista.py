# -*- coding: utf-8 -*-

import unittest

import numpy as np
import numpy.testing as npt

from ...tools import FISTA


class Test_FISTA(unittest.TestCase):

    def setUp(self):
        """
        """
        self.Nit = 1000
        # First optimizer performs arg min_x {x^2} with x scalar.
        # Nit is set to 1000.
        #
        self.optim1 = FISTA.FISTA(
            f=lambda x: x**2/2,
            df=lambda x: x,
            L=1,
            g=lambda x: 0,
            pg=lambda x: x,
            shape=(1,),
            Nit=self.Nit)

        # Second optimizer performs arg min_x {x^2} with x scalar.
        # Nit is not set.
        #
        self.optim2 = FISTA.FISTA(
            f=lambda x: x**2/2,
            df=lambda x: x,
            L=1,
            g=lambda x: 0,
            pg=lambda x: x,
            shape=(1,))

        # Third optimizer performs arg min_x {x^2} with x a 2-numpy array.
        # Nit is set to 1000.
        #
        self.optim3 = FISTA.FISTA(
            f=lambda x: (x[0]**2 + x[1]**2)/2,
            df=lambda x: x,
            L=1,
            g=lambda x: 0,
            pg=lambda x: x,
            shape=(2,),
            Nit=self.Nit)

    def test_StopCritera(self):
        """
        """
        # Tests StopCritera is None in case E is allclose to 0.
        #

        # Right now, self.optim1.E is np.zeros(Nit_max).
        cri = self.optim1.StopCritera(2)
        self.assertIsNone(cri)

        # Tests the computed value in case result is not None.
        #

        # Temp get E.
        Etmp = self.optim1.E

        # Sets E to another value.
        self.optim1.E = np.array([1, 2, 3, 4])
        cri = self.optim1.StopCritera(2)
        cri_2 = 1/self.optim1.tau

        npt.assert_allclose(cri, cri_2)

        # Put Etmp back.
        self.optim1.E = Etmp

    def test_StopTest(self):
        """
        """
        # Test in case Nit is set.
        #
        self.assertTrue(self.optim1.StopTest(0))
        self.assertTrue(self.optim1.StopTest(self.Nit-1))
        self.assertFalse(self.optim1.StopTest(self.Nit))

        # In case Nit is not set, should be true as long as n is str.
        # less than 2.
        #
        self.assertTrue(self.optim2.StopTest(0))
        self.assertTrue(self.optim2.StopTest(1))

        # In case Nit is not set, should be False for n g.t. Nit_max
        #
        self.assertFalse(self.optim2.StopTest(self.optim2.Nit_max))

        # In case Nit is not set, if E is close enough to 0, should be
        # False
        #
        self.assertFalse(self.optim2.StopTest(2))

        # In case Nit is not set, returns StopCritera > lim.
        #
        Etmp = self.optim2.E

        # StopCritera greater than lim: the critera for n=2 is 1.
        self.optim2.E = np.array([1, 2, 3, 4])
        self.assertTrue(self.optim2.StopTest(2))

        # StopCritera lower than lim: the critera for n=2 is 0.
        self.optim2.E = np.array([1, 1, 1, 1])
        self.assertFalse(self.optim2.StopTest(2))

        # Put Etmp back.
        self.optim2.E = Etmp

    def test_fista_dim_1(self):
        """First simple test for FISTA.

        Optimization problem is arg min_x {x^2} with x scalar.
        """
        #
        # With Nit
        xhat, InfoOut = self.optim1.execute()

        # Check output
        self.assertEqual(InfoOut['E'].size, self.Nit)
        self.assertGreater(InfoOut['time'], 0)
        npt.assert_allclose(xhat, 0)

        #
        # Without Nit
        xhat, InfoOut = self.optim2.execute()

        # Check that num_it is the pont where the stop test passes from
        # True to False.
        num_it = InfoOut['E'].size
        self.assertTrue(self.optim2.StopTest(num_it-1))
        self.assertFalse(self.optim2.StopTest(num_it))

        self.assertGreater(self.optim2.StopCritera(num_it-1),
                           self.optim2.lim)

        critera = self.optim2.StopCritera(num_it)
        if critera is not None:
            self.assertLess(critera, self.optim2.lim)
        else:
            npt.assert_almost_equal(self.optim2.E[-2], 0)

        # Check output
        self.assertGreater(InfoOut['time'], 0)
        self.assertLess(xhat, 5e-6)

    def test_fista_dim_2(self):
        """Second simple test for FISTA.

        Optimization problem is arg min_x {x^2} with x a 2-numpy array.
        """
        #
        # With Nit
        xhat, InfoOut = self.optim3.execute()

        # Check output
        npt.assert_allclose(xhat, np.array([0, 0]))
