# -*- coding: utf-8 -*-

import unittest
import tempfile
import os
import copy

import numpy as np
import numpy.testing as npt

import hyperspy.api as hs

from .. import signals


class Test_Path(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        self.shape = (3, 3)
        self.P = self.shape[0] * self.shape[1]
        self.path = np.array([3, 5, 7, 1])
        self.mask = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])
        self.scan = signals.Scan(self.shape, self.path)

    def test_get_mask(self):
        """
        """

        # Check that the generated mask is correct
        #
        mask = self.scan.get_mask()
        npt.assert_allclose(self.mask, mask)

    def test_ratio(self):
        # Check that the computed ratio is correct
        #
        npt.assert_allclose(self.scan.ratio, self.path.size / self.P)

        # Test ratio is one in case of full sampling.
        #
        scan2 = signals.Scan(self.shape, np.arange(self.P))
        npt.assert_allclose(scan2.ratio, 1)

        # Check that if the requires ratio is too high, the maximum
        # possible ratio is set.
        #
        scan3 = signals.Scan(self.shape, self.path, ratio=0.9)
        npt.assert_allclose(scan3.ratio, self.path.size / self.P)

        # Check that if the requires ratio is smaller, it is correctly
        # set.
        #
        scan4 = signals.Scan(self.shape, self.path, ratio=2/self.P)
        npt.assert_allclose(scan4.ratio, 2/self.P)

    def test_pixels_twice_in_path(self):
        # If a pixel index appears twice, check it calls a ValueError.
        #
        path = np.array([3, 3])
        with self.assertRaises(ValueError):
            signals.Scan(self.shape, path)

    def test_random(self):

        # Check that random with ratio is None leads to full sampling.
        #
        scan0 = signals.Scan.random(self.shape)
        npt.assert_allclose(scan0.ratio, 1)

        # Test that seed works
        #
        scan_a = signals.Scan.random(self.shape, ratio=0.5, seed=0)
        scan_b = signals.Scan.random(self.shape, ratio=0.5, seed=0)
        npt.assert_equal(scan_a.get_mask(), scan_b.get_mask())

    def test_from_file(self):

        # In case file ext is npz.
        #
        _, p = tempfile.mkstemp(suffix='.npz', text=True)

        m, n = self.shape
        dico = {'m': m, 'n': n, 'path': self.path}
        np.savez(p, **dico)

        scan_p = signals.Scan.from_file(p)

        npt.assert_equal(scan_p.path, self.path)
        npt.assert_equal(scan_p.ratio, self.path.size / self.P)

        os.remove(p)


class Test_search_nearest(unittest.TestCase):

    def test_several_cases(self):

        # Small test
        #
        mask = np.array([[0, 0, 0],
                         [1, 0, 1],
                         [0, 0, 0]])
        pos = signals.search_nearest(3, mask)
        npt.assert_equal(pos, ([1], [2]))

        # Small test #2
        #
        mask = np.array([[0, 0, 1],
                         [1, 0, 1],
                         [0, 0, 1]])
        pos = signals.search_nearest(3, mask)
        npt.assert_equal(pos, ([0, 1, 2], [2, 2, 2]))

        # Small test #3
        #
        mask = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [0, 0, 0]])
        pos = signals.search_nearest(3, mask)
        self.assertIsNone(pos)


class Test_Stem2D(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        # Simple data
        self.np_array = np.array([[0, 0, 0],
                                  [0, 1, 1],
                                  [1, 1, 1]], dtype=float)
        # Acquisition path
        self.path = np.array([3, 5, 1, 7])
        self.scan = signals.Scan(self.np_array.shape, self.path)

        # Create two acquisition objects
        self.obj_stem = signals.Stem2D(
            hs.signals.Signal2D(self.np_array),
            scan=self.scan)

    def test_correct_rows_removed(self):
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct(rows=slice(1, 2))

        npt.assert_allclose(
            local_obj.hsdata.data,
            np.array([[0, 1, 1]])
            )

    def test_correct_cols_removed(self):
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct(cols=slice(1, 2))

        npt.assert_allclose(
            local_obj.hsdata.data,
            np.array([[0], [1], [1]])
            )

    def test_correct_dead_pixels(self):
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct(dpixels=[3])
        expected_result = np.array([[0,   0,   0],
                                    [1/2, 1,   1],
                                    [1,   1,   1]])

        npt.assert_allclose(
            local_obj.hsdata.data,
            expected_result
            )

    def test_correct_non_sampled_dead_pixels(self):
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct(dpixels=[4])
        expected_result = np.array([[0,   0,   0],
                                    [0,   1,   1],
                                    [1,   1,   1]])

        npt.assert_allclose(
            local_obj.hsdata.data,
            expected_result
            )

    def test_correct_rows_and_dead_pixels(self):
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct(rows=slice(1, 3), dpixels=[3])
        expected_result = np.array([[1,   1,   1],
                                    [1,   1,   1]])

        npt.assert_allclose(
            local_obj.hsdata.data,
            expected_result
            )

        # Check that the sampled pixel at index 1 was removed from path.
        npt.assert_equal(local_obj.scan.path_0, np.array([0, 2, 4]))

    def test_correct_from_file(self):

        # Check that non-.conf files raise ValueError.
        #
        with self.assertRaises(ValueError):
            self.obj_stem.correct_fromfile('test.py')

        # Create the file.
        #
        # Temp file code
        _, confFile = tempfile.mkstemp(suffix='.conf', text=True)

        # Config.
        conf = """[2D DATA]
        rows = 1:3
        dpixels = [3]
        """

        # Fill config file.
        with open(confFile, 'w') as file:
            file.write(conf)

        # Ckeck it works
        #
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct_fromfile(confFile)
        expected_result = np.array([[1,   1,   1],
                                    [1,   1,   1]])

        npt.assert_allclose(
            local_obj.hsdata.data,
            expected_result
            )

        # Delete temp file.
        os.remove(confFile)

    def test_restore(self):
        pass


class Test_Stem3D(unittest.TestCase):
    """
    """
    def setUp(self):
        """
        """
        # Simple data
        self.np_array = np.array([[[0, 0, 0],
                                   [0, 1, 1],
                                   [1, 1, 1]],
                                  [[0, 0, 0],
                                   [0, 1, 1],
                                   [1, 1, 1]],
                                  [[0, 0, 0],
                                   [0, 1, 1],
                                   [1, 1, 1]]], dtype=float)
        # Acquisition path
        self.path = np.array([3, 5, 7, 1])
        self.scan = signals.Scan(self.np_array.shape[:2], self.path)

        # Create two acquisition objects
        self.obj_stem = signals.Stem3D(
            hs.signals.Signal1D(self.np_array),
            scan=self.scan)

    def test_correct_rows_removed(self):
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct(rows=slice(1, 2))
        expected_result = np.array(
            [[[0, 0, 0],
              [0, 1, 1],
              [1, 1, 1]]], dtype=float)

        npt.assert_allclose(
            local_obj.hsdata.data,
            expected_result
            )

    def test_correct_cols_removed(self):
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct(cols=slice(1, 2))
        expected_result = np.array(
            [[[0, 1, 1]],
             [[0, 1, 1]],
             [[0, 1, 1]]], dtype=float)

        npt.assert_allclose(
            local_obj.hsdata.data,
            expected_result
            )

    def test_correct_dead_pixels(self):
        local_obj = copy.copy(self.obj_stem)
        local_obj.correct(dpixels=[3])
        expected_res = np.array(
            [[[0, 0, 0],
              [0, 1, 1],
              [1, 1, 1]],
             [[0, 1, 1],
              [0, 1, 1],
              [1, 1, 1]],
             [[0, 0, 0],
              [0, 1, 1],
              [1, 1, 1]]], dtype=float)

        npt.assert_allclose(
            local_obj.hsdata.data,
            expected_res
            )

    def test_correct_rows_and_dead_pixels(self):
        local_obj = copy.copy(self.obj_stem)

        local_obj.correct(rows=slice(1, 3), dpixels=[3])
        expected_res = np.array(
            [[[0, 1, 1],
              [0, 1, 1],
              [1, 1, 1]],
             [[0, 0, 0],
              [0, 1, 1],
              [1, 1, 1]]], dtype=float)

        npt.assert_allclose(
            local_obj.hsdata.data,
            expected_res
            )

    def test_restore(self):
        pass
