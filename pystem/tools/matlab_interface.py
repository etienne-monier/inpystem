#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines an interface to run matlab codes from python.
"""

import os
import time
import sys
import logging
import pathlib

import numpy as np
import scipy.io as sio

_logger = logging.getLogger(__name__)


def matlab_interface(program, dataDico):
    """Interfaces a matlab code with python3.

    The functions needs a **matlab program** to run and **input data**
    to be given to the matlab program.

    The input data should be given in dictionary format where keys are
    the matlab variable names and values are the variable data.

    Arguments
    ---------
    program: str, Path object
        The program path.
    dataDico: dict
        The dico containing the data to give to the program.

    Returns
    -------
    dict
        The data returned by the program.

    Note
    ----
    A matlab command `matlab` should be accessible in the command line
    to make this code work.

    If this does not work, please be sure the PATH variable is perfecty
    set. For exemple, please add this to your `.bashrc` for Linux Users:

    .. code-block:: bash
      :caption: .bashrc

        export PATH:$PATH:/path/to/matlab/bin

    and for Windows users, please have a search about how to add a
    location into your path (this is a graphical task).
    """
    _logger.info('Preparing matlab interface.')

    # Check that the program is a matlab file.
    if isinstance(program, str):
        program = pathlib.Path(program)

    if program.suffix != '.m':
        raise ValueError(
            'The program is not a matlab file. Its extension is'
            ' {} instead of .m.'.format(program.suffix))

    program = program.resolve()

    # Get program directory name.
    progDirName = program.parent

    # Get data exchange directory
    dataDir = progDirName / 'InOut'
    dataDir.mkdir(parents=True, exist_ok=True)  # In case this dir does
    # not exist, create it.

    # Get data in and out files names
    dateStr = time.strftime('%A-%d-%B-%Y-%Hh%Mm%Ss', time.localtime())
    inName = dataDir / 'in_{}.mat'.format(id(dateStr))
    outName = dataDir / 'out_{}.mat'.format(id(dateStr))

    # Linux program
    #
    if sys.platform.startswith('linux'):

        # Give outname to program
        dataDico['outName'] = str(outName)

        # Save temp data.
        _logger.info('Saving Matlab data for interface.')
        sio.savemat(str(inName), dataDico)

        # Run code in matlab.
        _logger.info('Lanching matlab.')
        os.system(
            "matlab -nodesktop -nosplash -nodisplay -sd"
            " '{}' -r 'load(\"{}\"); run {}; quit;' | tail -n +11"
            .format(progDirName, inName, program.name))

        # Loads output data
        _logger.info('Loading Matlab results.')
        data = sio.loadmat(str(outName))

        # Erase temp input/output data
        _logger.info('Cleaning temporary files.')
        os.remove(str(inName))
        os.remove(str(outName))

    # Windows code
    #
    elif sys.platform.startswith('win32'):

        # Convert to windows path
        inName = pathlib.PureWindowsPath(inName)
        outName = pathlib.PureWindowsPath(outName)
        inName = pathlib.PureWindowsPath(inName)
        progDirName = pathlib.PureWindowsPath(progDirName)
        program = pathlib.PureWindowsPath(program)

        # Give outname to program
        dataDico['outName'] = str(outName)

        # Save temp data.
        _logger.info('Saving Matlab data for interface.')
        sio.savemat(str(inName), dataDico)

        # Run code in matlab.
        _logger.info('Lanching matlab.')
        os.system(
            "matlab -nodesktop -nosplash -sd"
            " '{}' -batch 'load(\"{}\"); run {}; quit;'"
            .format(progDirName, inName, program.name))

        # Loads output data
        _logger.info('Loading Matlab results.')
        data = sio.loadmat(str(outName))

        # Erase temp input/output data
        _logger.info('Cleaning temporary files.')
        os.remove(str(inName))
        os.remove(str(outName))

    else:
        _logger.error(
            "Sorry, we don't currently support Matlab interface for the " +
            sys.platform + " OS")

    # All output numpy data are squeezed, just in case ...
    for key, value in data.items():
        if type(value) is np.ndarray:
            data[key] = np.squeeze(value)

    return data
