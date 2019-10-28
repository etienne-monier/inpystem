# -*- coding: utf-8 -*-

import unittest

from ...tools import sec2str

params = [('0', '0s'),
          ('5', '5s'),
          ('5.2056', '5.21s'),
          ('10', '10s'),
          ('65', '1m 5s'),
          ('3905', '1h 5m 5s')]


class Test_sec2str(unittest.TestCase):
    def test_sec2str(self):
        for p1, p2 in params:
            with self.subTest(p1=p1, p2=p2):
                self.assertEqual(sec2str.sec2str(eval(p1)), p2)
