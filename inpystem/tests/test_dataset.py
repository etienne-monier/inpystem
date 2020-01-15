# -*- coding: utf-8 -*-

import unittest
import pathlib
import configparser
import os

import numpy as np
import numpy.testing as npt

from .. import dataset


class Test_Data_Path(unittest.TestCase):
    """
    """
    def setUp(self):

        self.p = pathlib.Path(__file__).parent / '..' / 'data' / 'path.conf'
        self.pbak = pathlib.Path(__file__).parent / '..' / 'data' /\
            'path.conf.back'

    def test_set_data_path(self):

        if self.p.exists():
            self.p.rename(self.pbak)

        try:
            # Test with a non-existing directory.
            #
            path = 'nom_bidon'
            self.assertFalse(dataset.set_data_path(path))

            # Test with a true directory
            #
            path = pathlib.Path(__file__).parent
            self.assertTrue(dataset.set_data_path(path))

            # Check the path has correctly been saved.
            #
            config = configparser.ConfigParser()
            config.read(self.p)
            self.assertEqual(config['Path']['data'], str(path))

        except AssertionError:
            # Be safe here not to lose the data path setting.
            #
            self.clear()
            raise

        else:
            self.clear()

    def test_read_data_path(self):

        if self.p.exists():

            # Read content.
            config = configparser.ConfigParser()
            config.read(self.p)
            path = config['Path']['data']

            # Make test.
            self.assertEqual(path, dataset.read_data_path())

        else:
            # Create config file
            path = pathlib.Path(__file__).parent
            dataset.set_data_path(str(path))

            # Make test.
            self.assertEqual(str(path), dataset.read_data_path())

            # Remove file
            os.remove(str(self.p))

    def clear(self):
        if self.pbak.exists():
            self.pbak.rename(self.p)
        else:
            os.remove(str(self.p))


# class Test_(unittest.TestCase):