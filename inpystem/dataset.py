# -*- coding: utf-8 -*-
"""This module defines important tools to import data from dataset.
"""

import pathlib
import logging
import configparser
import re

import numpy as np
import hyperspy.api as hs

from . import dev as devm
from . import signals as sig

_logger = logging.getLogger(__name__)


def read_data_path():
    """Read the saved data folder path.

    The inpystem library proposes to store all data in a particular
    directory with associated configuration files. This folder
    is saved in inpystem. To access to this folder path, use this
    function.

    If no data path is saved, the function returns None. Else, the
    path is returned.

    Returns
    -------
    None, str
        None is returned if no path is saved. Else, the data path
        is returned.
    """
    # This is the file which stores the data path.
    file_path = pathlib.Path(__file__).parent / 'path.conf'

    if file_path.exists():

        # The file exists. Let's read it.
        #
        config = configparser.ConfigParser()
        config.read(file_path)

        if 'Path' in config and 'data' in config['Path']:
            return config['Path']['data']

        # If the program gets here, that means that the file is
        # corrupted.
        _logger.error(
            'The dataset PATH file is corrupted.'
            'Please define one new with '
            'inpystem.dataset.set_data_path.'
            )
    else:
        # If the program gets here, that means that no data path is
        # saved.
        _logger.warning(
            'No dataset PATH has been defined. Please define one with '
            'inpystem.dataset.set_data_path.')

    return None


def set_data_path(path):
    """Sets the saved data folder path.

    The inpystem library proposes to store all data in a particular
    directory with associated configuration files. This folder
    is saved in inpystem. To set to this folder path, use this
    function.

    A boolean is returned to confirm that the change is effective.

    Arguments
    ---------
    path: str
        The desired data path.

    Returns
    -------
    bool
        If the data path has really been changed, the function
        returns True. Else, it returns False.
    """
    # We create a pathlib path from input.
    if isinstance(path, str):
        p = pathlib.Path(path)
    else:
        p = path

    # This is the file that stores the data path.
    file_path = pathlib.Path(__file__).parent / 'path.conf'

    if p.is_dir():

        # Input is correct and the code saves it.
        config = configparser.ConfigParser()
        config['Path'] = {}
        config['Path']['data'] = str(p.resolve())

        with open(file_path, 'w') as configfile:
            res = config.write(configfile)
    else:
        # The input path is invalid. Nothing is saved.
        _logger.warning(
            '{} is not a directory.'.format(str(p)))
        res = 0

    return res != 0


def load_file(
        file, ndim, scan_ratio=None, scan_seed=None, dev=None, verbose=True):
    """This function loads a STEM acquisition based on a configuration
    .conf file path.

    The number of dimensions ndim should also be given.

    The Path is generated from a scan file given in the configuration
    file or is randomly drawn. Whatever the case, the Scan object
    :code:`ratio` property can be set through the :code:`scan_ratio`
    argument. Additionally, in the case where no file is provided for
    the scan pattern, use the :code:`scan_seed` argument to have
    reproductible data.

    The function allows the user to ask for development data by setting
    the :code:`dev` argument. If :code:`dev` is None, then the usual
    Stem2D and Stem3D classes are returned. If :code:`dev` is a
    dictionary, then Dev2D and Dev3D classes are returned. This
    dictionary could contain additional class arguments such as:

    * snr, seed and normalized for Dev2D,
    * snr, seed, normalized, PCA_transformed and PCA_th for Dev3D.

    Arguments
    ---------
    file: str
        The configuration file path.
    ndim: int
        The data dimension. Should be 2 or 3.
    scan_ratio: optional, None, float
        The Path object ratio. Default is None for full sampling.
    scan_seed: int
        The seed in case of random scan initialization.
        Default is None for random seed.
    dev: optional, None, dictionary
        This arguments allows the user to ask for development data.
        If this is None, usual data is returned. If this argument is
        a dictionary, then development data will be returned and the
        dictionary will be given to the data contructors.
        Default is None for usual data.
    verbose: optional, bool
        If True, information will be sent to standard output..
        Default is True.

    Returns
    -------
    Stem2D, Stem3D, Dev2D, Dev3D
        The inpystem data.

    Todo
    ----
    Maybe enable PCA_th in config file for 3D data.
    """

    # Else, define DIR
    DIR = file.parent

    if verbose:
        print('Reading configuration file ...')

    # Read config file
    config = configparser.ConfigParser(
        converters={'list': lambda text: eval(text)})
    config.read(file)
    _logger.info('Read configuration file: {}.'.format(str(file)))

    # DATA
    #

    # Check that 2/3D DATA section (which is the only one to be
    # mandatory) is in the conf file
    section = '{}D DATA'.format(ndim)
    if section not in config:
        raise ValueError(
            'Mandatory section {}D DATA is not in config file.'.format(ndim))
    if 'file' not in config['{}D DATA'.format(ndim)]:
        raise ValueError('Data file not defined for {}D DATA.'.format(ndim))

    # Get data
    data_file = DIR / config[section]['file']

    if data_file.suffix == '.npy':

        # This is a numpy file
        ndata = np.load(str(data_file))
        if ndim == 2:
            data = hs.signals.Signal2D(ndata)
        else:
            data = hs.signals.Signal1D(ndata)

        # Some metadata are filled.
        data.metadata.original_filename = data_file.name
        data.metadata.title = data_file.stem

        # We search for some axis information in the configuration file.
        infos = ['name', 'scale', 'unit', 'offset']
        if 'AXES' in config:
            for dim in range(ndata.ndim):
                for info in infos:
                    info_key = 'axis_{}_{}'.format(dim, info)
                    if info_key in config['AXES']:

                        # Here, we've gone deep.
                        # We now have some info in the AXES section of
                        # the config file which have key info_key.
                        #
                        # This info can be a string or a float / int.
                        # Before saving it, let us get the value and
                        # know if this is really a string or not.
                        #
                        # For this, we look at the first letter of
                        # value. If that's a number, the value is
                        # evaluated.
                        value = config['AXES'][info_key]
                        if re.match('[0-9]', value[0]) is not None:
                            value = eval(value)
                        setattr(data.axes_manager[dim], info, value)

    else:
        data = hs.load(str(data_file))

    # SCAN
    #

    # Get sampling scan
    if 'SCAN' in config and 'file' in config['SCAN']:
        scan_file = DIR / config['SCAN']['file']
        scan = sig.Scan.from_file(scan_file, ratio=scan_ratio)
    else:
        scan = sig.Scan.random(
            data.data.shape[:2], ratio=scan_ratio, seed=scan_seed)

    # Data object
    #
    if verbose:
        print('Generating data ...')

    if dev is None:

        _logger.info('Generating {}D data from file.'.format(ndim))

        if ndim == 2:
            obj = sig.Stem2D(data, scan, verbose)
        else:
            obj = sig.Stem3D(data, scan, verbose)

        # Correct data from file.
        #
        obj.correct_fromfile(file)

    else:

        _logger.info('Generating {}D dev data from file.'.format(ndim))

        key = file.stem
        if ndim == 2:
            obj = devm.Dev2D(
                key, data, modif_file=file, scan=scan, **dev, verbose=verbose)
        else:
            obj = devm.Dev3D(
                key, data, modif_file=file, scan=scan, **dev, verbose=verbose)

    return obj


def load_key(
        key, ndim, scan_ratio=None, scan_seed=None, dev=None, verbose=True):
    """This function loads a STEM acquisition based on a key.

    A key is a string which can be:

    * an example data name,
    * the name of some data located in the inpystem data path (which is
      defined with the :py:func:`inpystem.dataset.set_data_path`
      function).

    The key should always be the name of the configuration file without
    the suffix (.conf). As an example, if a configuration file located
    in the data folder is named my-sample.conf, then its data could be
    loaded with the my-sample key.

    The number of dimensions ndim should also be given.

    The Path is generated from a scan file given in the configuration
    file or is randomly drawn. Whatever the case, the Scan object
    :code:`ratio` property can be set through the :code:`scan_ratio`
    argument. Additionally, in the case where no file is provided for
    the scan pattern, use the :code:`scan_seed` argument to have
    reproductible data.

    The function allows the user to ask for development data by setting
    the :code:`dev` argument. If :code:`dev` is None, then the usual
    Stem2D and Stem3D classes are returned. If :code:`dev` is a
    dictionary, then Dev2D and Dev3D classes are returned. This
    dictionary could contain additional class arguments such as:

    * snr, seed, normalized and verbose for Dev2D,
    * snr, seed, normalized, PCA_transformed, PCA_th and verbose for
      Dev3D.

    This function only searches for the configuration file to use the
    load_file function afterwards.

    Arguments
    ---------
    key: str
        The data key.
    ndim: int
        The data dimension. Should be 2 or 3.
    scan_ratio: optional, None, float
        The Path object ratio. Default is None for full sampling.
    scan_seed: int
        The seed in case of random scan initialization.
        Default is None for random seed.
    dev: optional, None, dictionary
        This arguments allows the user to ask for development data.
        If this is None, usual data is returned. If this argument is
        a dictionary, then development data will be returned and the
        dictionary will be given to the data contructors.
        Default is None for usual data.
    verbose: optional, bool
        If True, information will be sent to standard output..
        Default is True.

    Returns
    -------
    Stem2D, Stem3D, Dev2D, Dev3D
        The inpystem data.
    """

    if read_data_path() is None:
        raise ImportError('The data directory is not defined.')

    if ndim != 2 and ndim != 3:
        raise ValueError('Invalid number of dimension. Should be 2 or 3.')

    # Let us set the directory where to search for the key.conf file.
    #

    # Let's have a look at the data path.
    data_path = pathlib.Path(read_data_path())

    # This small part searches for the file key.conf in the data_path
    # directory. If this is found, the flag is set to True and file is
    # the path of the config file.
    if verbose:
        print('Searching for key {}...'.format(key), end=' ')

    found_flag = False
    for file in data_path.rglob("*.conf"):
        if file.stem == key:
            found_flag = True
            break

    # If the key is not valid, a ValueError is raised.
    if not found_flag:
        if verbose:
            print('')
        raise ValueError('Invalid key {} for dataset.'.format(key))

    if verbose:
        print('found')

    _logger.info('Loading data from key: {}.'.format(key))

    return load_file(file, ndim, scan_ratio, scan_seed, dev, verbose)
