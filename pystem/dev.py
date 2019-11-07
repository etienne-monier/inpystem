#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the basic stem acquisitions objects used for processing.
These objects are:

1. The 3D spectrum-image,
2. The 2D HAADF image,
"""

import abc

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import hyperspy.api as hs

from .tools import PCA
from . import signals as sig


class AbstractDev(abc.ABC):
    """Abstract Dev acquisition class.

    This is an *abstract* class, which mean you can not instantiate such
    object.

    It defines the structure for a Dev acquisition object.

    key: str
        1-word description of the Dev2D image.
    data: (m,n) or (m, n, l) numpy array
        The Dev2D image data before the noise step.
        Its dimension is (m,n).
    ndata: (m,n) or (m, n, l) numpy array
        The noised Dev2D image.
        If :code:`snr` is None, :code:`ndata` is None.
        Its dimension is (m,n).
    sigma: float
        The noise standard deviation.
    seed: optional, int
        The random noise matrix seed.
    normalize: bool
        If :code:normalize` is True, the data will be centered
        and normalize before the corruption steps.
    mean_std: None, 2-tuple
        It stores the data mean and std in case normalize is True.
    verbose: bool
        If True, information will be displayed.
        Default is True.
    """
    def __init__(self, key, data, mask=None, sigma=None, seed=None,
                 normalize=True, verbose=True):
        """AbstractDev constructor.

        Arguments
        ---------
        key: str
            1-word description of the Dev2D image.
            Generally, it's common to the stem acquisition object.
        data: (m, n) or (m, n, l) numpy array
            The noise-free image data.
        mask: (m, n) numpy array
            The sampling mask.
        sigma: optional, None, float
            The desired standard deviation used to model noise.
            Dafault is None for no additional noise.
        seed: optional, None, int
            The random noise matrix seed.
            Dafault is None for no seed initialization.
        normalize: optional, bool
            If :code:normalize` is True, the data will be centered
            and normalize before the corruption steps.
            Default is True.
        verbose: optional, bool
            If True, information will be displayed.
            Default is True.
        """

        # Save inputs
        self.key = key
        self.data = data
        self.ndata = None

        self.normalize = normalize
        self.verbose = verbose

        # Normalize if necessary
        self.mean_std = None

        if self.normalize:

            # The sampled data position.
            y, x = np.nonzero(mask)
            # Correct data.
            correct_data = self.data[y, x] if data.ndim == 2 else \
                self.data[y, x, :]
            mean = correct_data.mean()
            std = correct_data.std()

            self.data = (self.data - mean) / std
            self.mean_std = (mean, std)

        # Setup noise
        self._sigma = sigma

        # sets seed
        self._seed = seed
        self.seed = seed

        if self._sigma is not None:

            # Update noised image
            self.set_ndata()

    @property
    def seed(self):
        """seed property getter.
        """
        return self._seed

    @seed.setter
    def seed(self, value):
        """seed property setter.
        """
        # Set seed value
        npr.seed(self._seed)

        # Update value
        self._seed = value

    @property
    def sigma(self):
        """sigma property getter.
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        """sigma property setter.

        This function modifies the noisy image standard deviation.

        Arguments
        ---------
        value: float
            The desired standard deviation.
        """
        # get sigma
        if value is not None:

            # Set new value
            self._sigma = value

            # Update noised image
            self.set_ndata()

        else:
            self.ndata = None
            self._sigma = None

    def set_ndata(self):
        """ Constructs the noised data.

        It is also used to draw a new noise matrix.
        """
        # Draw noise
        noise_matrix = npr.randn(*self.data.shape)

        # Compute noised data.
        self.ndata = self.data + self.sigma * noise_matrix


class Dev2D(sig.Stem2D, AbstractDev):
    """Dev2D Class.

    Attributes
    ----------
    key: str
        1-word description of the Dev2D image.
    hsdata: Signal2D hyperspy data
        The hyperspy Signal2D image.
        Its dimension is denoted (m,n).
        This is used to communicate with the parrent class.
    data: (m,n) numpy array
        The Dev2D image data before the noise step.
        Its dimension is (m,n).
    ndata: (m,n) numpy array
        The noised Dev2D image.
        If :code:`snr` is None, :code:`ndata` is None.
        Its dimension is (m,n).
    scan : optional, Scan object
        The sampling scan object associated with the data.
        Default is None for full sampling.
    sigma: float
        The noise standard deviation.
    seed: optional, int
        The random noise matrix seed.
    normalize: bool
        If :code:normalize` is True, the data will be centered
        and normalize before the corruption steps.
    mean_std: None, 2-tuple
        It stores the data mean and std in case normalize is True.
    verbose: bool
        If True, information will be displayed.
        Default is True.
    """

    def __init__(
            self, key, hsdata, scan=None, modif_file=None, sigma=None,
            seed=None, normalize=True, verbose=True):
        """SpectrumImage constructor.

        Arguments
        ---------
        key: str
            1-word description of the Dev2D image.
            Generally, it's common to the stem acquisition object.
        hsdata: Signal2D hyperspy data
            The noise-free Dev2D image data.
            Its dimension is denoted (m,n).
        scan : optional, None, Scan object
            The sampling scan object associated with the data.
            Default is None for full sampling.
        modif_file: optional, None, str
            A .conf configuration file to remove rows, columns or dead
            pixels. Default is None for no modification.
        sigma: optional, None, float
            The desired standard deviation used to model noise.
            Dafault is None for no additional noise.
        seed: optional, None, int
            The random noise matrix seed.
            Dafault is None for no seed initialization.
        normalize: optional, bool
            If :code:normalize` is True, the data will be centered
            and normalize before the corruption steps.
            Default is True.
        verbose: optional, bool
            If True, information will be displayed.
            Default is True.
        """
        sig.Stem2D.__init__(self, hsdata, scan, verbose)

        # Checks is modification is required
        if modif_file is not None:
            self.correct_fromfile(modif_file)

        AbstractDev.__init__(
            self, key, self.hsdata.data, self.scan.get_mask(), sigma, seed,
            normalize, verbose)

        # Updates hs data
        self.hsdata.data = self.data if self._sigma is None else self.ndata

    def restore(self, method='interpolation', parameters={}, verbose=None):
        """
        """
        if verbose is None:
            verbose = self.verbose

        # Updates hs data
        self.hsdata.data = self.data if self._sigma is None else self.ndata
        return sig.Stem2D.restore(self, method, parameters, verbose)

    def plot(self, noised=False):
        """ Plots the haadf image.

        Arguments
        ---------
        noised: optional, bool
            If True, the noised data is used.
            If False, the noise-free data is shown.
            Default is False.
        """

        if noised and self.ndata is None:
            raise ValueError(
                'Can not display noised data when snr has been set to None.')

        # Show data
        hs_tmp = hs.signals.Signal2D(self.data if not noised else self.ndata)
        hs_tmp.axes_manager = self.hsdata.axes_manager
        hs_tmp.metadata = self.hsdata.metadata
        hs_tmp.plot()

    def __repr__(self):
        """function __repr__
        """
        L = sig.Stem2D.__repr__(self)[1:-1].split(',')
        L[0] = 'Dev2D'
        L2 = ','.join(L)
        return '<{}>'.format(L2)


class Dev3D(sig.Stem3D, AbstractDev):
    """Dev3D Class

    Attributes
    ----------
    key: str
        1-word description of the Dev3D.
    hsdata: Signal1D hyperspy data
        The hyperspy Signal2D image.
        Its dimension is denoted (m,n).
        This is used to communicate with the parrent class.
    data: (m,n,l) numpy array
        The Dev3D data before the noise step.
        Its dimension is (m,n,l).
    ndata: (m,n,l) numpy array
        The noised Dev3D data.
        If snr is None, ndata is None. Its dimension is (m,n,l).
    snr: optional, float
        The desired snr used for the noising step.
    sigma: float
        The noise standard deviation.
    seed: optional, int
        The random noise matrix seed.
    normalize: bool
        If normalize is True, the data will be centered and normalize
        before the corruption steps.
    PCA_transform: bool
        If PCA_transformed is True, a PCA transformation has
        been applied to the data.
    PCA_info: None, dictionary
        If PCA_transformed is True, PCA_info contains informations about
        the reduction. Otherwise, it is None.
    PCA_operator: PcaHandler
        The PCA operator.
    verbose: bool
        If True, information will be displayed.
        Default is True.
    """

    def __init__(
            self, key, hsdata, scan=None, modif_file=None, sigma=None,
            seed=None, normalize=True, PCA_transform=False, PCA_th='auto',
            verbose=True):
        """Dev3D __init__ function.

        Arguments
        ---------
        key: str
            1-word description of the Dev3D image.
            Generally, it's common to the stem acquisition object.
        hsdata: Signal1D hyperspy data
            The noise-free Dev3D image data.
            Its dimension is denoted (m,n, l).
        scan : optional, None, Scan object
            The sampling scan object associated with the data.
            Default is None for full sampling.
        modif_file: optional, None, str
            A .conf configuration file to remove rows, columns or dead
            pixels. Default is None for no modification.
        sigma: optional, None, float
            The desired standard deviation used to model noise.
            Dafault is None for no additional noise.
        seed: optional, None, int
            The random noise matrix seed.
            Dafault is None for no seed initialization.
        normalize: optional, bool
            If :code:normalize` is True, the data will be centered
            and normalize before the corruption steps.
            Default is True.
        PCA_transform: optional, bool
            If PCA_transformed is True, a PCA transformation is applied
            to the data.
            Default is False.
        PCA_th: optional, str, int
            The desired data dimension after dimension reduction.
            Possible values are:

            * 'auto' for automatic choice,
            * 'max' for maximum value
            * an int value for user value.

            Default is 'auto'.
        verbose: optional, bool
            If True, information will be displayed.
            Default is True.
        """
        sig.Stem3D.__init__(self, hsdata, scan, verbose)

        # Checks is modification is required
        if modif_file is not None:
            self.correct_fromfile(modif_file)

        # Apply PCA if required
        self.PCA_operator = PCA.PcaHandler(
            self.hsdata.data, mask=self.scan.get_mask(),
            PCA_transform=PCA_transform, PCA_th=PCA_th, verbose=verbose)

        self.PCA_transform = PCA_transform
        data = self.PCA_operator.Y_PCA
        self.PCA_info = self.PCA_operator.InfoOut

        # Initialization as AbstractDev object.
        AbstractDev.__init__(
            self, key, data, self.scan.get_mask(), sigma, seed,
            normalize, verbose)

        # Updates hs data
        self.hsdata.data = self.data if self._sigma is None else self.ndata
        # In case the data is in the PCA space
        if self.PCA_transform:
            self.PCA_info['axis_name'] = self.hsdata.axes_manager[2].name
            self.hsdata.axes_manager[2].name = 'PCA'
            self.hsdata.axes_manager[2].size = self.data.shape[-1]

    def direct_transform(self, data):
        """Applies the Dev3D PCA transformation and normalization
        steps to data.

        Arguments
        ---------
        data: (m, n, l) numpy array, hs image
            Data whose shape is the same as self.data.
        """
        if type(data) is not np.ndarray:
            # Data comes from hs.
            data2 = data.data
            hsflag = True
        else:
            # Data comes from numpy
            data2 = data
            hsflag = False

        # Apply PCA and normalization
        #
        data3 = self.PCA_operator.direct(data2)

        # If the data is normalize, normalize also.
        if self.normalize:
            data3 = (data3 - self.mean_std[0]) / self.mean_std[1]

        # If hs data, copy axes manager and metadata
        #
        if hsflag:
            data.data = data3
            if self.PCA_transform:
                data.axes_manager[2].name = 'PCA'
                data.axes_manager[2].size = data3.shape[-1]
            return data
        return data3

    def inverse_transform(self, data):
        """ Applies the Dev3D PCA inverse transformation and
        inverse normalization steps to spim.

        Arguments
        ---------
        data: (m, n, l) numpy array, hs image
            Data whose shape is the same as self.data.
        """
        if type(data) is not np.ndarray:
            # Data comes from hs.
            data2 = data.data
            hsflag = True
        else:
            # Data comes from numpy
            data2 = data
            hsflag = False

        if self.normalize:
            data2 = data2 * self.mean_std[1] + self.mean_std[0]

        data3 = self.PCA_operator.inverse(data2)

        if hsflag:
            data.data = data3
            if self.PCA_transform:
                data.axes_manager[2].name = self.PCA_info['axis_name']
                data.axes_manager[2].size = data3.shape[-1]
            return data
        return data3

    def restore(self, method='interpolation', parameters={},
                PCA_transform=None, PCA_th='auto', verbose=None):
        """
        """
        if verbose is None:
            verbose = self.verbose
        if PCA_transform is None:
            PCA_transform = not self.PCA_transform

        # Updates hs data
        self.hsdata.data = self.data if self._sigma is None else self.ndata

        return sig.Stem3D.restore(
            self, method, parameters, PCA_transform, PCA_th, verbose)

    def show_sum(self, noised=False):
        """ Shows the sum of the data along the last axis.

        Arguments
        ---------
        noised: optional, bool
            If True, the noised data is used.
            If False, the noise-free data is shown.
            Default is False.
        """

        if noised and self.ndata is None:
            raise ValueError(
                'Can not display noised data when snr has been set to None.')

        # Show data
        fig, ax = plt.subplots()

        if not noised:
            ax.matshow(self.data.sum(2), cmap='viridis')
        else:
            ax.matshow(self.ndata.sum(2), cmap='viridis')

        # set title
        ax.set_title("{}: sum over energy loss axis".format(self.key))

        # layout
        ax.axis('off')

    def plot_as2D(self, noised=False):
        """Implements the HypersSpy tool to visualize the image for a given band.

        Arguments
        ---------
        noised: optional, bool
            If True, the noised data is used.
            If False, the noise-free data is shown.
            Default is False.
        """

        if noised and self.ndata is None:
            raise ValueError(
                'Can not display noised data when snr has been set to None.')

        if noised:
            hs_data = hs.signals.Signal2D(
                np.moveaxis(self.ndata, [0, 1, 2], [1, 2, 0]))
        else:
            hs_data = hs.signals.Signal2D(
                np.moveaxis(self.data, [0, 1, 2], [1, 2, 0]))

        hs_data.axes_manager = self.hsdata.axes_manager
        hs_data.metadata = self.hsdata.metadata

        hs_data.plot(cmap='viridis')

    def plot_as1D(self, noised=False):
        """Implements the HypersSpy tool to visualize the spectrum for a given pixel.

        Arguments
        ---------
        noised: optional, bool
            If True, the noised data is used.
            If False, the noise-free data is shown.
            Default is False.
        """

        if noised and self.ndata is None:
            raise ValueError(
                'Can not display noised data when snr has been set to None.')

        if noised:
            hs_data = hs.signals.Signal1D(self.ndata)
        else:
            hs_data = hs.signals.Signal1D(self.data)

        hs_data.axes_manager = self.hsdata.axes_manager
        hs_data.metadata = self.hsdata.metadata

        hs_data.plot()
        hs_data._plot.navigator_plot.ax.images[-1].set_cmap("viridis")

    def plot_roi(self, noised=False):
        """Implements the Hyperspy tool to analyse regions of interest.

        Arguments
        ---------
        noised: optional, bool
            If True, the noised data is used.
            If False, the noise-free data is shown.
            Default is False.
        """

        if noised and self.ndata is None:
            raise ValueError(
                'Can not display noised data when snr has been set to None.')

        # Create Hyperspy data
        if noised:
            hs_data = hs.signals.Signal1D(self.ndata)
        else:
            hs_data = hs.signals.Signal1D(self.data)

        hs_data.axes_manager = self.hsdata.axes_manager
        hs_data.metadata = self.hsdata.metadata

        # Create ROI
        roi = hs.roi.RectangularROI(left=0, top=0, right=5, bottom=5)

        # Plot Signal
        hs_data.plot()
        hs_data._plot.navigator_plot.ax.images[-1].set_cmap("viridis")

        # Creates an interactively sliced Signal
        roi = roi.interactive(hs_data)

        # Computes the mean over the ROI
        mean_roi = hs.interactive(
            roi.mean, event=roi.axes_manager.events.any_axis_changed)

        # Plot ROI
        mean_roi.plot()
        # Sets the color of the roi 2D plot
        # mean_roi._plot.navigator_plot.ax.images[-1].set_cmap("viridis")

    def __repr__(self):
        """function __repr__
        """
        L = sig.Stem3D.__repr__(self)[1:-1].split(',')
        L[0] = 'Dev3D'
        L2 = ','.join(L)
        return '<{}>'.format(L2)
