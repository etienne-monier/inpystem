#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines an interface to run matlab codes from python.
"""

import os
import time
import sys
from pathlib import PureWindowsPath

import scipy.io as sio


def matlab_interface(program, dataDico):
    """Interfaces a matlab code with python3.

    The functions needs a **matlab program** to run and **input data**
    to be given to the matlab program.

    The input data should be given in dictionary format where keys are
    the matlab variable names and values are the variable data.

    Arguments
    ---------
    program: Path object
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

    # Check that the program is a matlab file.
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
    # not exist.

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
        sio.savemat(str(inName), dataDico)

        # Run code in matlab.
        os.system(
            "matlab -nodesktop -nosplash -nodisplay -sd"
            " '{}' -r 'load(\"{}\"); run {}; quit;' | tail -n +11"
            .format(progDirName, inName, program.name))

        # Loads output data
        data = sio.loadmat(str(outName))

        # Erase temp input/output data
        os.remove(str(inName))
        os.remove(str(outName))

    # Windows code
    #
    elif sys.platform.startswith('win32'):

        # Convert to windows path
        inName = PureWindowsPath(inName)
        outName = PureWindowsPath(outName)
        inName = PureWindowsPath(inName)
        progDirName = PureWindowsPath(progDirName)
        program = PureWindowsPath(program)

        # Give outname to program
        dataDico['outName'] = str(outName)

        # Save temp data.
        sio.savemat(str(inName), dataDico)

        # Run code in matlab.
        os.system(
            "matlab -nodesktop -nosplash -sd"
            " '{}' -batch 'load(\"{}\"); run {}; quit;'"
            .format(progDirName, inName, program.name))

        # Loads output data
        data = sio.loadmat(str(outName))

        # Erase temp input/output data
        os.remove(str(inName))
        os.remove(str(outName))

    else:
        print(
            "Sorry, we don't currently have support for the " +
            sys.platform + "OS")

    return data
