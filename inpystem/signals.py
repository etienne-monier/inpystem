# -*- coding: utf-8 -*-
"""This package defines all sort of classes to handle data for inpystem.
"""

import abc
import pathlib
import configparser
import logging
import copy

import numpy as np
import matplotlib.pyplot as plt
import hyperspy.api as hs

from . import restore
from .tools import misc

_logger = logging.getLogger(__name__)


def search_nearest(pix, mask):
    """Searches the non-zeros pixels of mask which are near to the pix
    pixel.

    The algorithm beggins by searching for non-zero elements around the
    pixel whose index is pix. If there are non-zero elements, their
    positions are returned. Otherwise, the searching horizon rises by
    one. Then, the algorithm goes on.

    The algorithm stops automatically at an horizon of 100 pixels. In
    such case, an error is sent to the logger and None is returned.

    Arguments
    ---------
    pix: int
        The position index of the pixel whose neighbors should be found.
    mask: (m, n) numpy array
        The sampling mask (1 for sampled pixel and 0 otherwise).

    Returns
    -------
    None, 2-tuple
        None is returned in case no neighbor was found. Otherwise, a
        tuple (y, x) is returned where x (resp. y) are the columns
        (resp. rows) index of the pixel neighbors
.    """

    m, n = mask.shape
    y, x = np.unravel_index(pix, (m, n))
    mask[y, x] = 0

    Lmax = 100
    flag = True
    L = 1

    pmask = np.pad(mask, pad_width=Lmax, mode='constant')
    py, px = y + Lmax, x + Lmax

    while flag:

        submask = pmask[py-L:py+L+1, px-L:px+L+1]
        if submask.sum() > 0:
            flag = False
        else:
            if L < Lmax:
                L += 1
            else:
                return None

    nnz = submask.nonzero()

    return (list(nnz[0] + (y-L)), list(nnz[1] + (x-L)))


class Scan:
    """Scan pattern class.

    This class stores the data spatial shape and two copies of the
    scan pattern. One of these copies is the initial scan pattern
    which is given to the class. At the same time, a :code:`ratio`
    argument can be given to keep only a portion of the available
    samples. See Notes for more details.


    Attributes
    ----------
    shape : 2-length tuple
        The spatial shape (m, n) where m is the number of rows and n is the
        number of columns.
    path : numpy array
        The sampling path to be used in the study.
    path_0 : numpy array
        The initial sampling path to be kept in case the ratio is changed.
    ratio: float
        The current :code:`ratio` value such that :code:`path`has size
        :code:`ratio*m*n`. Changing this attribute automaticaly updates
        :code:`path`.

    Note
    -----
    Consider only :code:`r*m*n` pixels hve been sampled, then the
    :code:`path_0` attribute has shape (r*m*n, ) and its elements
    lay between 0 and m*n-1.

    Meanwhile, if the user wants to consider only :code:`ratio` percent of
    the samples, the :code:`ratio` argument should be given. The
    :code:`path` attribute would then have shape (ratio*m*n, ). In such
    case, :code:`path_0[:ratio*m*n]` will be equal to
    :code:`path`. Be aware that :code:`ratio` should be lower than
    :code:`r`.

    Each element of these arrays is the pixel index in row major order.
    To recover the row and column index array, type the following commands.

    ::code:
        i = path // n
        j = path % n
    """

    def __init__(self, shape, path, ratio=None):
        """Scan pattern constructor.

        Arguments
        ---------
        shape: (m, n) tuple
            The spatial shape where m is the number of rows and n is
            the number of columns.
        path : tuple, numpy array
            The sampling path. See class Notes for more detail.
        ratio: optional, float
            The ratio of sampled pixels. This should lay between 0 (excl.)
            and 1. Default is None for full sampling.
        """
        if len(tuple(shape)) != 2:
            raise ValueError('Invalid shape parameter length.')

        path = np.asarray(path)
        if path.ndim != 1:
            raise ValueError('Input path array should have 1 dimension.')
        if path.size == 0 or path.size > shape[0] * shape[1]:
            raise ValueError('Input path array has invalid size.')

        path = path.astype(int)

        # Let us check that no value is given twice
        path_copy = path.copy()
        path_copy.sort()

        if np.any(path_copy[1:] - path_copy[:-1] == 0):
            raise ValueError(
                'Some elements of input path appear at least '
                'twice.')

        self.shape = shape
        self.path = path
        self.path_0 = path

        self._ratio = ratio

        # The following line initializes self.path
        self.ratio = ratio

    @property
    def ratio(self):
        """ Ratio getter.

        Returns
        -------
        float
            The ratio property value.
        """
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        """ Ratio setter.

        It checks that the value is correct and updates the scan
        object attributes.

        Arguments
        ---------
        value: float
            The new ratio value.
        """

        if value is not None and (value > 1 or value <= 0):
            raise ValueError('The ratio should be in the ]0,1] segment.')

        # get sample size
        m, n = self.shape

        if value is not None:
            N = int(m * n * value)  # Number of required samples.
        else:
            N = self.path_0.size
            self._ratio = N / (m*n)

        if N > self.path_0.size:
            N = self.path_0.size
            _logger.warning(
                'Input ratio is higher than higher maximal ratio'
                ' ({:.3f}). Ratio is set to the maximal value.'.format(
                    self.path_0.size/(m*n)))

        # change scan pattern length
        self.path = self.path_0[:N]

        # Set new value
        self._ratio = N / (m*n)

    @classmethod
    def from_file(cls, data_file, ratio=None):
        """ Creates a scan pattern object from a data file
        (such as .dm3, .dm4 or npz).

        In the case of a .npz file, this one should contain the :code:̀'m`,
        :code:̀'n` and :code:̀'path` variables which are resp. the number of
        rows and columns and the path array.

        Concerning the .dm3/.dm4 files, the data storage is specific to
        the LPS Lab (Orsay, France) implementation.

        An aditional argument :code:̀'ratio` allows you to select only a
        given ratio of the sampled pixels. This should lay between 0 (excl.)
        and 1.

        Arguments
        ---------
        data_file: str
            The data file path.
        ratio: optional, float
            The ratio of sampled pixels. This should lay between 0 (excl.)
            and 1. Default is None for full sampling.

        Returns
        -------
        Scan object
            The scan pattern.
        """
        _logger.info('Loading Scan file.')

        # Get file extension
        p = pathlib.Path(data_file)
        file_ext = p.suffix

        # Digital Micrograph file
        if (file_ext == '.dm3' or file_ext == '.dm4'):
            _logger.info('Scan file type is {}.'.format(file_ext))

            data = hs.load(str(p)).data

            if data.ndim == 3:
                _logger.info('{} scan file type is A.'.format(file_ext))
                m, n, _ = data.shape
                x = data[:, :, 0].flatten()-1
                y = data[:, :, 1].flatten()-1

            elif data.ndim == 2:
                _logger.info('{} scan file type is B.'.format(file_ext))
                m, n = data.shape
                x = data.flatten() % n
                y = data.flatten() // n

            else:
                raise ValueError(
                    'Scan data has {} dimensions. Expected dimension'
                    ' is 2 or 3.'.format(data.ndim)
                    )
            path = y * n + x

        # Numpy file
        elif (file_ext == '.npz'):
            _logger.info('Scan file type is .npz.')

            data = np.load(str(data_file))
            m, n = int(data['m']), int(data['n'])
            path = data['path']

        return cls((m, n), path, ratio=ratio)

    @classmethod
    def random(cls, shape, ratio=None, seed=None):
        """ Creates a random scan pattern object.

        Arguments
        ---------
        shape: (m, n) tuple
            The data spatial shape.
        ratio: optional, float
            The ratio of sampled pixels. It should lay between 0 (excluded)
            and 1. Default is None for full sampling.
        seed: optional, int
            Seed for random sampling.
            Default is None for random seed.

        Returns
        -------
        Scan object
            The scan pattern.
        """
        _logger.info('Random scan generated.')

        if seed is not None:
            np.random.seed(seed)

        # The following code should do the job.
        #
        # P = shape[0]*shape[1]  # Number of pixels.
        # perm = np.random.permutation(P)
        #
        # However, to match previous version output for seed 0, the
        # following non-optimal code is chosen.
        pix_ratio = 1 if ratio is None else ratio
        mask = np.random.rand(*shape) < pix_ratio

        perm = np.flatnonzero(mask)
        np.random.shuffle(perm)

        return cls(shape, perm)

    def get_mask(self):
        """Returns the sampling mask.

        The sampling mask is boolean and True is for sampled pixels.

        Returns
        -------
        mask : (m, n) numpy array
            The sampling mask.
        """
        mask = np.zeros(self.shape, dtype=bool)

        sampled_pos = np.unravel_index(self.path, self.shape)
        mask[sampled_pos] = True

        return mask

    def plot(self):
        """ Plots the sampling mask.

        White (resp. black) pixels are sampled (resp. non-sampled).
        """

        # Show data
        fig, ax = plt.subplots()
        ax.matshow(self.get_mask())
        # set title
        ax.set_title('Sampling mask')
        # layout
        ax.axis('image')
        ax.axis('off')

    def __repr__(self):
        return "<Scan, shape: {}, ratio: {:.3f}>".format(
            self.shape, self.ratio)


class AbstractStem(abc.ABC):
    """Abstract STEM acquisition class.

    This is an *abstract* class, which mean you can not instantiate such
    object.

    It defines the structure for a STEM acquisition object.

    Attributes
    ----------
    hsdata : hs BaseSignal
        The acquired STEM data hyperspy object.
    scan : Scan object
        The sampling scan object associated with the data.
    verbose: bool
        If True, information is sent to standard output.
        Default is True.
    """
    def __init__(self, hsdata, scan=None, verbose=True):
        """
        AbstractStem constructor.

        Arguments
        ---------
        hsdata : hs BaseSignal
            The acquired STEM data hyperspy object.
        scan : optional, Scan object
            The sampling scan object associated with the data.
            Default is None for full sampling.
        verbose: bool
            If True, information is sent to standard output.
            Default is True.
        """
        if verbose:
            print('Creating STEM acquisition...')

        if issubclass(type(hsdata), hs.signals.BaseSignal):
            self.hsdata = hsdata.copy()
        else:
            raise ValueError('Invalid input data.')

        if scan is None:
            scan = Scan.random(hsdata.data.shape)

        if scan.shape != hsdata.data.shape[:2]:
            raise ValueError('hsdata and scan have incoherent spatial shape.')

        self.scan = copy.deepcopy(scan)
        self.verbose = verbose

    def correct(
            self, rows=slice(None), cols=slice(None), bands=slice(None),
            dpixels=None):
        """Correct deffective data.

        Deffective data correspond to:

        1. Rows to remove at the begging or at the end of the image.
        2. Columns to remove at the begging or at the end of the image.
        3. Bands to remove at the begging or at the end of the image.
        4. Located dead pixels at the center of the image.

        In the cases 1, 2 or 3, the rows and columns are purely removed.
        The dead pixels are filled with the mean over a neighbourhood.

        A :code`slice` object for an object :code:`A` of length :code:`L`
        defines a continuous portion of :code:`A` such as
        :code:`A[n_1], A[n_1+1], ..., A[n_2-1]`
        with :code:`n_1` < :code:`n_2`. In such case, a slice object
        definition is :code:`slice(n_1, n_2)`. If :code:`n_1` is 0, then
        use :code:`slice(None, n_2)`. If :code:`n_2` is :code:`L`
        use :code:`slice(n_1, None)`. Last, if all the elements of
        :code:`A` should be kept, use :code:`slice(None)`.

        Arguments
        ---------
        rows: slice object
            The range of rows to keep.
        cols: slice object
            The range of columns to keep.
        cols: slice object
            The range of bands to keep.
        dpixels: list
            The positions of the dead pixels.
        """

        if self.verbose:
            print('Correcting STEM acquisition...')

        _logger.info('Correcting STEM data.')

        # Get Numpy data
        data = self.hsdata.data
        m, n = data.shape[:2]

        if dpixels is not None:

            # Preparing the sampling mask before searching for neighbor
            # pixels.
            #

            # Valid path array with non-dead sampled pixels.
            path = self.scan.path_0[
                np.isin(self.scan.path_0, dpixels, invert=True)
                ]
            # Position of these non-dead sampled pixels.
            valid_pos = np.unravel_index(path, (m, n))

            mask0 = np.zeros((m, n))
            mask0[valid_pos] = 1

            # Remove rows/columns that will be removed afterwards.
            mask = np.zeros((m, n))
            mask[rows, cols] = mask0[rows, cols]

            for dpix in dpixels:

                if np.sum(np.isin(self.scan.path_0, dpixels)) == 0:
                    # The dead pixel were not sampled.
                    _logger.warning(
                        'The dead pixel at position {} was not sampled and '
                        'was ignored for correction.'.format(dpix))
                    continue

                # Search for the neighbors.
                pixels = search_nearest(dpix, mask)

                if pixels is None:
                    _logger.error(
                        'No sampled pixel were found near enough '
                        'to the dead pixel at position {}. Correction '
                        'for this pixel was aborted.'.format(dpix))
                else:
                    y, x = np.unravel_index(dpix, mask.shape)

                    if data.ndim == 2:
                        data[y, x] = np.mean(data[pixels[0], pixels[1]])
                    else:
                        data[y, x, :] = np.mean(
                            data[pixels[0], pixels[1], :], axis=0)

        # Remove missing columns and rows.
        #

        if cols.step is not None or rows.step is not None:
            raise ValueError(
                'Limit slices should have a None as a step.')
        if data.ndim == 3 and bands.step is not None:
            raise ValueError(
                'Limit slices should have a None as a step.')

        if data.ndim == 2:
            data = data[rows, cols]
        else:
            data = data[rows, cols, bands]

        m, n = data.shape[:2]

        # Handle axes_manager
        #

        # Getting initial axes_manager
        tmp_axes_manager = self.hsdata.axes_manager

        # Modifying limits
        tmp_axes_manager[0].size = n
        tmp_axes_manager[1].size = m
        if data.ndim == 3:
            tmp_axes_manager[2].size = data.shape[2]

        # Modify eV axis scale and offset.
        if data.ndim == 3:
            if bands.start is not None:
                offset = tmp_axes_manager[2].offset
                scale = tmp_axes_manager[2].scale
                tmp_axes_manager[2].offset = offset + bands.start * scale

        # Defining new hs data object.
        im_tmp = hs.signals.Signal2D(data) if data.ndim == 2 else\
            hs.signals.Signal1D(data)
        # Adding modified axes_manager.
        im_tmp.axes_manager = tmp_axes_manager

        # Handle metadata
        #
        im_tmp.metadata = self.hsdata.metadata

        # Setting it as new data in self.
        self.hsdata = im_tmp

        # Handle scan pattern
        #

        p0 = self.scan.path_0       # Initial scan path.
        ratio = self.scan.ratio     # scan ratio.
        m, n = self.scan.shape      # Initial data shape.
        m1, n1 = data.shape[:2]     # New data shape.

        # Getting limits in old axis
        xmin = cols.start if cols.start is not None else 0
        xmax = xmin + n1
        ymin = rows.start if rows.start is not None else 0
        ymax = ymin + m1

        # Getting the old axis y and x position of sampled pixels.
        y0, x0 = np.unravel_index(p0, (m, n))
        # Getting p mask of sampled pixels that belong to the new axis.
        ind = np.flatnonzero(
            np.logical_and(
                np.logical_and(y0 >= ymin, y0 < ymax),
                np.logical_and(x0 >= xmin, x0 < xmax)
            ))

        # The new sampling path.
        p_new = (y0[ind] - ymin) * n1 + (x0[ind] - xmin)

        # The new scan object.
        self.scan = Scan((m1, n1), p_new, ratio)

    def correct_fromfile(self, file):
        """
        """

        # The path of the config file is defined.
        p = pathlib.Path(file)

        if p.suffix != '.conf':
            raise ValueError('Input file extension should be .conf.')

        # The config file is read.
        config = configparser.ConfigParser(
            converters={'list': lambda text: eval(text)})
        config.read(p)

        # The section inside which to look at.
        ndim = self.hsdata.data.ndim
        section = '{}D DATA'.format(ndim)

        keys = ['rows', 'columns', 'bands']
        slices = []
        for cnt in range(ndim):
            value = config[section].get(keys[cnt], fallback=None)
            slices.append(
                misc.toslice(value, length=self.hsdata.data.shape[cnt]))

        dpixels = config[section].getlist('dpixels', fallback=None)

        self.correct(*slices, dpixels=dpixels)

    @abc.abstractmethod
    def restore(self):
        """Restores corrupted data.
        """
        pass

    def plot(self):
        """Plots the masked data.
        """
        hs_tmp = self.hsdata.copy()

        if hs_tmp.data.ndim == 2:
            hs_tmp.data = hs_tmp.data * self.scan.get_mask()
        else:
            B = self.hsdata.data.shape[-1]
            hs_tmp.data = hs_tmp.data * \
                np.tile(
                    self.scan.get_mask()[:, :, np.newaxis],
                    [1, 1, B])
        hs_tmp.plot()

    def __repr__(self):

        L = self.hsdata.__repr__()[1:-1].split(',')
        L[0] = 'Stem{}D'.format(self.hsdata.data.ndim)
        L.append(' sampling ratio: {:.2f}'.format(self.scan.ratio))
        L2 = ','.join(L)
        return '<{}>'.format(L2)


class Stem2D(AbstractStem):
    """2D image STEM acquisition.

    This defines a 2D STEM image with its associated sampling scan.

    Attributes
    ----------
    hsdata : hs BaseSignal
        The acquired STEM data hyperspy object.
    scan : Path object
        The sampling scan object associated with the data.
    """
    def restore(self, method='interpolation', parameters={}):
        """Restores the acquisition.

        It performs denoising in the case of full scan and performs
        recontruction in case of partial sampling.

        Arguments
        ---------
        """
        if self.verbose:
            print('Restoring the 2D STEM acquisition...')

        data = self.hsdata.data

        if self.scan is not None:
            mask = self.scan.get_mask()
        else:
            mask = np.ones(data.shape, dtype=bool)

        _logger.info('Restoration with method {} was intended.'.format(
            method))

        method_dico = {
            'INTERPOLATION': restore.interpolation.interpolate,
            'L1': restore.LS_2D.L1_LS,
            'ITKRMM': restore.DL_ITKrMM.ITKrMM,
            'WKSVD': restore.DL_ITKrMM.wKSVD,
            'ITKRMM_MATLAB': restore.DL_ITKrMM_matlab.ITKrMM_matlab,
            'WKSVD_MATLAB': restore.DL_ITKrMM_matlab.wKSVD_matlab,
            'BPFA_MATLAB': restore.DL_BPFA.BPFA_matlab
        }

        method_u = method.upper()
        if method_u in method_dico:

            xhat, InfoOut = method_dico[method_u](
                Y=data, mask=mask, verbose=self.verbose, **parameters)

        else:
            raise ValueError('Unknown method {}.'.format(method))

        xhat_hs = hs.signals.Signal2D(xhat)
        xhat_hs.axes_manager = self.hsdata.axes_manager
        xhat_hs.metadata = self.hsdata.metadata

        return xhat_hs, InfoOut


class Stem3D(AbstractStem):
    """3D image STEM acquisition.

    This defines a 3D STEM image with its associated sampling scan.

    Attributes
    ----------
    hsdata : hs BaseSignal
        The acquired STEM data hyperspy object.
    scan : Path object
        The sampling scan object associated with the data.
    """
    def restore(self, method='interpolation', parameters={},
                PCA_transform=True, PCA_th='auto'):
        """Restores the acquisition.

        It performs denoising in the case of full scan and performs
        recontruction in case of partial sampling.

        Arguments
        ---------

        PCA_transform: optional, bool
            Enables the PCA transformation if True, otherwise, no PCA
            transformation is processed.
            Default is True.
        PCA_th: optional, int, str
            The desired data dimension after dimension reduction.
            Possible values are 'auto' for automatic choice, 'max' for maximum
            value and an int value for user value.
            Default is 'auto'.
        """
        if self.verbose:
            print('Restoring the 3D STEM acquisition...')

        data = self.hsdata.data

        if self.scan is not None:
            mask = self.scan.get_mask()
        else:
            mask = np.ones(data.shape[:2], dtype=bool)

        _logger.info('Restoration with method {} was intended.'.format(
            method))

        method_dico = {
            'INTERPOLATION': restore.interpolation.interpolate,
            '3S': restore.LS_3D.SSS,
            'SNN': restore.LS_3D.SNN,
            'CLS': restore.LS_CLS.CLS,
            'CLS_POST_LS': restore.LS_CLS.Post_LS_CLS,
            'ITKRMM': restore.DL_ITKrMM.ITKrMM,
            'WKSVD': restore.DL_ITKrMM.wKSVD,
            'ITKRMM_MATLAB': restore.DL_ITKrMM_matlab.ITKrMM_matlab,
            'WKSVD_MATLAB': restore.DL_ITKrMM_matlab.wKSVD_matlab,
            'BPFA_MATLAB': restore.DL_BPFA.BPFA_matlab
        }

        method_u = method.upper()
        if method_u in method_dico:

            xhat, InfoOut = method_dico[method_u](
                Y=data, mask=mask, verbose=self.verbose,
                PCA_transform=PCA_transform, PCA_th=PCA_th,
                **parameters)

        else:
            raise ValueError('Unknown method {}.'.format(method))

        Xhat_hs = hs.signals.Signal1D(xhat)
        Xhat_hs.axes_manager = self.hsdata.axes_manager
        Xhat_hs.metadata = self.hsdata.metadata

        return Xhat_hs, InfoOut

    def plot_sum(self):
        """ Shows the sum of the data along the last axis.
        """

        # Show data
        fig, ax = plt.subplots()
        ax.matshow(self.hsdata.data.sum(2) * self.scan.get_mask())

        # set title
        ax.set_title("{}: sum over last axis".format(
            self.hsdata.metadata.General.title))

        # layout
        ax.axis('off')

    def plot_as2D(self):
        """Implements the HypersSpy tool to visualize the image for a
        given band.
        """
        mask3 = np.tile(
            self.scan.get_mask()[:, :, np.newaxis],
            [1, 1, self.hsdata.data.shape[2]])
        hs_data = hs.signals.Signal2D(
                np.moveaxis(self.hsdata.data * mask3, [0, 1, 2], [1, 2, 0]))

        axis_0 = [0, 1, 2]
        axis_1 = [2, 0, 1]
        for n in range(3):
            hs_data.axes_manager[axis_0[n]].name = \
                self.hsdata.axes_manager[axis_1[n]].name
            hs_data.axes_manager[axis_0[n]].offset = \
                self.hsdata.axes_manager[axis_1[n]].offset
            hs_data.axes_manager[axis_0[n]].scale = \
                self.hsdata.axes_manager[axis_1[n]].scale
            hs_data.axes_manager[axis_0[n]].units = \
                self.hsdata.axes_manager[axis_1[n]].units

        hs_data.metadata = self.hsdata.metadata

        hs_data.plot()

    def plot_as1D(self):
        """Implements the HypersSpy tool to visualize the spectrum for
        a given pixel.
        """
        mask3 = np.tile(
            self.scan.get_mask()[:, :, np.newaxis],
            [1, 1, self.hsdata.data.shape[2]])
        hs_data = hs.signals.Signal1D(self.hsdata.data * mask3)

        hs_data.axes_manager = self.hsdata.axes_manager
        hs_data.metadata = self.hsdata.metadata

        hs_data.plot()

    def plot_roi(self):
        """Implements the Hyperspy tool to analyse regions of interest.
        """
        # Create Hyperspy data
        hs_data = hs.signals.Signal1D(self.hsdata.data)

        hs_data.axes_manager = self.hsdata.axes_manager
        hs_data.metadata = self.hsdata.metadata

        # Create ROI
        roi = hs.roi.RectangularROI(left=0, top=0, right=5, bottom=5)

        # Plot Signal
        hs_data.plot()

        # Creates an interactively sliced Signal
        roi = roi.interactive(hs_data)

        # Computes the mean over the ROI
        mean_roi = hs.interactive(
            roi.mean, event=roi.axes_manager.events.any_axis_changed)

        # Plot ROI
        mean_roi.plot()
