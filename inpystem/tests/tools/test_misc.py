# -*- coding: utf-8 -*-

import unittest

from ...tools import misc


class Test_misc(unittest.TestCase):

    def test_toslice(self):
        """
        """
        # Case 5:
        s = misc.toslice('5:')
        self.assertEqual(s.start, 5)
        self.assertIsNone(s.stop)
        self.assertIsNone(s.step)

        # Case :10
        s = misc.toslice(':10')
        self.assertIsNone(s.start)
        self.assertEqual(s.stop, 10)
        self.assertIsNone(s.step)

        # Case 5:10
        s = misc.toslice('5:10')
        self.assertEqual(s.start, 5)
        self.assertEqual(s.stop, 10)
        self.assertIsNone(s.step)

        # Case :-10
        N = 100
        s = misc.toslice(':-10', length=N)
        self.assertIsNone(s.start)
        self.assertEqual(s.stop, N-10)
        self.assertIsNone(s.step)

        # Case 5:-10
        s = misc.toslice('5:-10', length=N)
        self.assertEqual(s.start, 5)
        self.assertEqual(s.stop, N-10)
        self.assertIsNone(s.step)

        # Case :N
        s = misc.toslice(':100', length=N)
        self.assertIsNone(s.start)
        self.assertEqual(s.stop, N)
        self.assertIsNone(s.step)
