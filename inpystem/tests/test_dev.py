# -*- coding: utf-8 -*-

import unittest

import numpy as np
import numpy.testing as npt

import hyperspy.api as hs

from .. import signals
from .. import dev


class Test_Dev2D(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        shape = (100, 100)

        # Simple data
        np.random.seed(0)
        self.np_array = np.random.rand(*shape)
        # Acquisition path
        self.scan = signals.Scan.random(shape, ratio=0.5)

        # hs data
        self.hsdata = hs.signals.Signal2D(self.np_array)

    def test_noise(self):

        sigma = 100
        key = 'test'

        # Check seed works
        self.obj_stem = dev.Dev2D(
            key,
            self.hsdata,
            scan=self.scan,
            sigma=sigma,
            seed=0)

        self.obj_stem2 = dev.Dev2D(
            key,
            self.hsdata,
            scan=self.scan,
            sigma=sigma,
            seed=0)

        # For noise-free data
        npt.assert_allclose(
            self.obj_stem.data,
            self.obj_stem2.data
            )
        # For noisy data
        npt.assert_allclose(
            self.obj_stem.ndata,
            self.obj_stem2.ndata
            )

        # Check sigma is OK.
        y, x = np.nonzero(self.scan.get_mask())
        npt.assert_allclose(
            (self.obj_stem.ndata - self.obj_stem.data)[y, x].std(),
            sigma,
            atol=3
            )

        # Check that drawing a new noise changes the noise
        noise1 = self.obj_stem.ndata - self.obj_stem.data
        self.obj_stem.set_ndata()
        noise2 = self.obj_stem.ndata - self.obj_stem.data

        self.assertFalse(np.allclose(noise1, noise2))

    def test_normalise(self):

        sigma = 100
        key = 'test'

        # Check normalisation works
        self.obj_stem = dev.Dev2D(
            key,
            self.hsdata,
            sigma=sigma,
            seed=0,
            normalize=True)

        npt.assert_allclose(self.obj_stem.data.mean(), 0, atol=1e-15)
        npt.assert_allclose(self.obj_stem.data.std(), 1)
