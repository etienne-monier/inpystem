# -*- coding: utf-8 -*-

import unittest
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
        self.size = (3, 3)
        self.path = np.array([3, 5, 7, 1])
        self.mask = np.array([[0, 1, 0],
                              [1, 0, 1],
                              [0, 1, 0]])
        self.path_obj = signals.Path(self.size, self.path)

    def test_get_mask(self):
        """
        """
        mask = self.path_obj.get_mask()
        npt.assert_allclose(self.mask, mask)


# class Test_Stem2D(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         """
#         # Simple data
#         self.np_array = np.array([[0, 0, 0],
#                                   [0, 1, 1],
#                                   [1, 1, 1]], dtype=float)
#         # Acquisition path
#         self.path = np.array([3, 5, 7, 1])

#         # Create two acquisition objects
#         self.obj_stem = signals.Stem2D(
#             hs.signals.Signal2D(self.np_array),
#             path=self.path)

#     def test_correct_rows_removed(self):
#         local_obj = copy.copy(self.obj_stem)
#         local_obj.correct(rows=(1, 2))

#         npt.assert_allclose(
#             local_obj.data.data,
#             np.array([[0, 1, 1]])
#             )

#     def test_correct_cols_removed(self):
#         local_obj = copy.copy(self.obj_stem)
#         local_obj.correct(cols=(1, 2))

#         npt.assert_allclose(
#             local_obj.data.data,
#             np.array([[0], [1], [1]])
#             )

#     def test_correct_dead_pixels(self):
#         local_obj = copy.copy(self.obj_stem)
#         local_obj.correct(dpixels=[3])
#         expected_result = np.array([[0,   0,   0],
#                                     [3/5, 1,   1],
#                                     [1,   1,   1]])

#         npt.assert_allclose(
#             local_obj.data.data,
#             expected_result
#             )

#     def test_correct_rows_and_dead_pixels(self):
#         local_obj = copy.copy(self.obj_stem)
#         local_obj.correct(rows=(1, 3), dpixels=[4])
#         expected_result = np.array([[0,   0.8, 1],
#                                     [1,   1,   1]])

#         npt.assert_allclose(
#             local_obj.data.data,
#             expected_result
#             )

#     def test_restore(self):
#         pass


# class Test_Stem3D(unittest.TestCase):
#     """
#     """
#     def setUp(self):
#         """
#         """
#         # Simple data
#         self.np_array = np.array([[[0, 0, 0],
#                                    [0, 1, 1],
#                                    [1, 1, 1]],
#                                   [[0, 0, 0],
#                                    [0, 1, 1],
#                                    [1, 1, 1]],
#                                   [[0, 0, 0],
#                                    [0, 1, 1],
#                                    [1, 1, 1]]], dtype=float)
#         # Acquisition path
#         self.path = np.array([3, 5, 7, 1])

#         # Create two acquisition objects
#         self.obj_stem = signals.Stem3D(
#             hs.signals.Signal1D(self.np_array),
#             path=self.path)

#     def test_correct_rows_removed(self):
#         local_obj = copy.copy(self.obj_stem)
#         local_obj.correct(rows=(1, 2))
#         expected_result = np.array(
#             [[[0, 0, 0],
#               [0, 1, 1],
#               [1, 1, 1]]], dtype=float)

#         npt.assert_allclose(
#             local_obj.data.data,
#             expected_result
#             )

#     def test_correct_cols_removed(self):
#         local_obj = copy.copy(self.obj_stem)
#         local_obj.correct(cols=(1, 2))
#         expected_result = np.array(
#             [[[0, 1, 1]],
#              [[0, 1, 1]],
#              [[0, 1, 1]]], dtype=float)

#         npt.assert_allclose(
#             local_obj.data.data,
#             expected_result
#             )

#     def test_correct_dead_pixels(self):
#         local_obj = copy.copy(self.obj_stem)
#         local_obj.correct(dpixels=[3])
#         expected_res = np.array(
#             [[[0, 0, 0],
#               [0, 1, 1],
#               [1, 1, 1]],
#              [[0, 0.6, 0.6],
#               [0, 1, 1],
#               [1, 1, 1]],
#              [[0, 0, 0],
#               [0, 1, 1],
#               [1, 1, 1]]], dtype=float)

#         npt.assert_allclose(
#             local_obj.data.data,
#             expected_res
#             )

#     def test_correct_rows_and_dead_pixels(self):
#         local_obj = copy.copy(self.obj_stem)

#         local_obj.correct(rows=(1, 3), dpixels=[3])
#         expected_res = np.array(
#             [[[0, 2/3, 2/3],
#               [0, 1, 1],
#               [1, 1, 1]],
#              [[0, 0, 0],
#               [0, 1, 1],
#               [1, 1, 1]]], dtype=float)

#         npt.assert_allclose(
#             local_obj.data.data,
#             expected_res
#             )

#     def test_restore(self):
#         pass
