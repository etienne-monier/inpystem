# -*- coding: utf-8 -*-

import unittest
import tempfile
import shutil

import numpy as np
import numpy.testing as npt

from ...tools import matlab_interface


class Test_matlab_interface(unittest.TestCase):

    def setUp(self):
        """
        """
        # Creating temporary dir and files.
        #

        # Temp. Folder
        tmpDir = tempfile.mkdtemp()

        # Temp file code
        _, codeFile = tempfile.mkstemp(dir=tmpDir, suffix='.m', text=True)

        # Matlab code.
        #
        # This small code just computes the square of the input matrix.
        #
        code = "B = A.^2; save(outName, 'B');"

        # Fill code file.
        with open(codeFile, 'w') as file:
            file.write(code)

        self.tmpDir = tmpDir
        self.codeFile = codeFile

    def test_matlab_interface(self):
        """This small test just check the interface with a Matlab code
        that performs a square operation.
        """
        # Input data
        A = np.arange(5)

        res = matlab_interface.matlab_interface(
            program=self.codeFile,
            dataDico={'A': A}
            )

        npt.assert_allclose(A**2, res['B'])

    def tearDown(self):
        """
        """
        # Delete all temp dir.
        shutil.rmtree(self.tmpDir)
