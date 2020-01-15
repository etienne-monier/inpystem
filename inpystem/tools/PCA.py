# -*- coding: utf-8 -*-
"""
This module implements tools to perform PCA transformation.

The main element is the **PcaHandler** class which is a user interface.
It performs direct and inverse PCA transformation for 3D data.

**Dimension_Reduction** is the background function which performs PCA
while the **EigenEstimate** function improves the estimation of PCA
eigenvalues.
"""

import time
import logging

import numpy as np

from . import sec2str

_logger = logging.getLogger(__name__)


def EigenEstimate(l, Ns):
    """ Computes an estimate of the covariance eigenvalues given the sample
    covariance eigenvalues. The Stein estimator coupled with isotonic
    regression has been used here.

    For more information, have a look at:

        *
        * MESTRE, Xavier. Improved estimation of eigenvalues and
          eigenvectors of covariance matrices using their sample
          estimates. IEEE Transactions on Information Theory, 2008,
          vol. 54, no 11, p. 5113-5129.s

    Arguments
    ---------
    l: numpy array
        Sample eigenvalues
    Ns: int
        Number of observations

    Returns
    -------
    numpy array
        Estimated covariance matrix eigenvalues.
    float
        Estimated Gaussian noise standard deviation.
    int
        Estimated dimension of the signal subspace.
    """

    if l.ndim != 1:
        raise ValueError('Input array l should have one dimension.')

    # Get data dimension
    M = l.size

    # Initial data ----------------------
    #
    # The initial data consists in a table
    #
    # +-----+-----+-----+---------+
    # | l_0 | l_1 | ... | l_{M-1} |
    # +-----+-----+-----+---------+
    # | a_0 | a_1 | ... | a_{M-1} |
    # +-----+-----+-----+---------+
    # where l (resp. a) are lattent variables (resp. denominator of Stein
    # estimator).

    # Stein estimator
    table = np.stack((l, np.zeros(l.size)), axis=0)

    for col in range(M):

        # That's an (M, )-array filled with 1 but at col position.
        ncol = np.logical_not(np.in1d(range(M), col))

        table[1, col] = 1 + (1 / M) * np.sum(
            (l[col] + l[ncol]) / (l[col] - l[ncol])
            )
        # table[1, col] = Ns - M + 1 + 2 * l[col] * np.sum(
        #     1/(l[col] - l[ncol])
        #     )

    # Procedure 1st step ----------------------
    #
    # Here, the goal is to make all a_i positive.
    #
    # 1. Start at the right of the table and search to the left until the
    #    first pair (l_j, a_j) with negative a_j is reached.
    # 2. Pool this pair with the pair imediately on the left of it,
    #    replacing them with the pair (l_j + l_{j-1}, a_j + a_{j-1}), to
    #    form a list which is on pair shorter.
    # 3. Repeat 1 and 2 until all a_j are positive.
    #
    # We will denote here M1 the length of the modified table.
    #
    # The back_pos variable is a list of lists. Its length will be M1 at the
    # end of this step. back_pos[j1] for j1 smaller than M1 will be all the
    # columns of the initial table that were used to create the column j1
    # of the new table.

    back_pos = [[i] for i in range(M)]

    while (np.any(table[1, :] < 0)):  # there are <0 alphai

        # Initial cursor position in the table.
        cpos = table.shape[1] - 1

        # Searching the position of the negative a_i.
        while (table[1, cpos] > 0):
            cpos = cpos - 1

        # The sum of the two pairs.
        sum_pairs = np.sum(table[:, cpos-1:cpos+1], axis=1)[:, np.newaxis]

        # Depending of the cases, the arrays to stack are different.
        if cpos == table.shape[1] - 1:  # That's the last pair.
            hstack_ar = (table[:, :cpos-1], sum_pairs)

        elif cpos == 1:  # That's the first pair
            hstack_ar = (sum_pairs, table[:, cpos+1:])

        else:  # The cursor is in the middle of table.
            hstack_ar = (table[:, :cpos-1], sum_pairs, table[:, cpos+1:])

        # Create new table
        table = np.hstack(hstack_ar)

        # Modify index list.
        back_pos[cpos-1].extend(back_pos[cpos])
        del back_pos[cpos]

    # Procedure 2nd step ----------------------
    #
    # Here, the goal is to re-order the ratios l_j/a_j so that they are
    # decreasing.
    #
    # To that end, a row will be added to table, which is the ratio of the
    # first and the second lines.
    #
    # A pair (l_j, a_j) is called violating pair if the ratio l_j/a_j is not
    # larger than l_{j+1}/a_{j+1}.
    #
    # 1. Start at the bottom of the list found in Step 1 and proceed to the
    #    left until the first violating pair, say (l_j, a_j), is reached.
    # 2. Pool this violating pair with the pair immediately on the right by
    #    replacing these two pairs and their ratios with the pair
    #    (l_j+l_{j+1}, a_j+a_{j+1}) and its ratio
    #    (l_j+l_{j+1})/(a_j+a_{j+1}), forming a new list shorter by one pair.
    # 3. Start at the pair imediately at the right (or the replacing pair
    #    itself if that's the last one) and proceed to the left until a
    #    violating pair is found, then repeat 2.
    # 4. Repeat 3 until all ratios l_j/a_j are in decreasing order.
    #
    # In this step, the back_pos variable will be modified in a similar way
    # as for Step 1.

    table = np.vstack((table, table[0, :] / table[1, :]))

    # Current position
    cpos = table.shape[1] - 2

    # If cpos get to -1, it means that no pair is violating.
    while cpos >= 0:

        while table[2, cpos+1] < table[2, cpos] and cpos >= 0:
            cpos = cpos - 1

        if cpos >= 0:
            # A violating pair was found.

            # The pairs are summed.
            sum_pairs = np.sum(table[:, cpos:cpos+2], axis=1)[:, np.newaxis]
            sum_pairs[2] = sum_pairs[0] / sum_pairs[1]

            # Depending of the cases, the arrays to stack are different.
            if cpos == table.shape[1] - 2:  # That's the before last pair.
                hstack_ar = (table[:, :cpos], sum_pairs)

            elif cpos == 0:  # That's the first pair
                hstack_ar = (sum_pairs, table[:, cpos+2:])

            else:  # The cursor is in the middle of table.
                hstack_ar = (table[:, :cpos], sum_pairs, table[:, cpos+2:])

            # Create new table
            table = np.hstack(hstack_ar)

            # Modify index list.
            back_pos[cpos].extend(back_pos[cpos+1])
            del back_pos[cpos+1]

            # Move the cursor to the left if cpos is at the extreme right.
            if cpos == table.shape[1] - 1:
                cpos = table.shape[1] - 2

    # Procedure 3nd step ----------------------
    #
    # Each ratio in the final table was obtained by pooling a block of one
    # or more consecutive pairs in the original list. To obtain Stein's
    # modified estimates, we assign this ratio to all pairs of the block.

    # Stein estimate output.
    sl = np.zeros(M)

    for cnt in range(table.shape[1]):
        sl[back_pos[cnt]] = table[2, cnt] * np.ones(len(back_pos[cnt]))

    # Sigma and dimension estimation ----------------------
    #

    # A threasholding is applied to avoid zero estimates.
    sl[sl < 1e-12] = 1e-12

    # Noise standard deviation is estimated to be the last Stein estimate.
    sig = np.sqrt(sl[-1])

    # The signal dimension is estimated to be the first positin such that sl
    # is equal to sig.
    D = min(back_pos[-1])

    return (sl, sig, D)


def Dimension_Reduction(Y, mask=None, PCA_th='auto', verbose=True):
    """Reduces the dimension of a multi-band image.

    Arguments
    ---------
    Y: (m, n, l) numpy array
        The multi-band image where the last axis is the spectral one.
    mask: optional, (m, n) numpy array
        The spatial sampling mask filled with True where pixels are sampled.
        This is used to remove correctly the data mean.
        Default if a matrix full of True.
    PCA_th: optional, str, int
        The PCA threshold.
        'auto' for automatic estimation.
        'max' to keep all components.
        An interger to choose the threshold.
        In case there are less samples (N) than the data dimension (l),
        thi sparameter is overridded to keep a threshold of N-1.
    verbose: optional, bool
        Prints output if True. Default is True.

    Returns
    -------
    (m, n, PCA_th) numpy array
        The data in the reduced subspace.
        Its shape is (m, n, PCA_th) where PCA_th is the estimated data
        dimension.
    dict
        The dictionary contaning additional information about the reduction.
        See Note.

    Note
    ----
    The InfoOut dictionary containg the thee following keys:

    1. 'H' which is the base of the reduced subspace.
       Its shape is (l, PCA_th) where PCA_th is the estimated data
       dimension.
    2. 'd' which is the evolution of the PCA-eigenvalues after estimation.
    3. 'PCA_th' which is the estimated data dimension.
    4. 'sigma' which is the estimated Gaussian noise standard deviation.
    5. 'Ym' which is a (m, n, l) numpy array where the data mean over bands
       is repeated for each spatial location.
    """
    if mask is not None and mask.shape != Y.shape[:2]:
        raise ValueError('Incoherent mask shape.')

    # Default mask is full sampling.
    #
    if mask is None:
        mask = np.ones(Y.shape[:2])

    # Start messaage
    #
    if verbose:
        print("- PCA transformation -")
    start = time.time()

    # Store the data dimensions.
    #
    m, n, M = Y.shape
    N = int(mask.sum())
    P = m * n

    # Remove data mean
    #

    # Reshape data and remove mean
    # Compute the indexes of the non-zeros elements of the flatten mask.
    nnz = np.flatnonzero(mask)
    Yr = Y.reshape((n * m, M)).T   # Reshaped data have 'r' additianal letter.
    # Compute the mean along bands of the reshaped data.
    Yrm = np.tile(np.mean(Yr[:, nnz], axis=1), (P, 1)).T
    Yrwm = Yr - Yrm  # Remove mean

    # Perform PCA.
    #
    [d, V] = np.linalg.eigh(np.cov(Yrwm[:, nnz]))
    ind = np.argsort(d)[::-1]
    d = d[ind]
    V = V[:, ind]

    # Selct only the N first elements in case less samples than dim.
    # N <= M
    #
    if (N <= M):
        _logger.warning('Number of samples is lower than data dimension.')
        d = d[:N - 1]
        V = V[:, :N - 1]

    # Perform Stein isotonic regression
    #
    _logger.info('Performing Stein regression.')
    dout, sigma, Rest = EigenEstimate(d, N)

    # Sets the PCA threshold level
    #
    if N <= M:
        Auto = np.minimum(N-1, Rest)
        Max = N - 1
    else:
        Auto = Rest
        Max = np.minimum(M, N)

    if PCA_th == 'auto':
        th = Auto

    elif PCA_th == 'max':
        th = Max

    elif PCA_th > Max:
        _logger.warning(
            'PCA threshold too high. '
            'Highest possible value used instead.')
        th = Max

    else:
        th = PCA_th

    th = int(th)
    _logger.info('Threshold is {}.'.format(th))

    # Prepare output.
    #
    H = V[:, :th]
    S = np.dot(H.T, Yrwm).T.reshape((m, n, th))
    Yrrm = Yrm.T.reshape((m, n, M))

    # Output message
    #
    if (verbose):
        print(
            'Dimension reduced from {} to {}.\n'
            'Estimated sigma^2 is {:.2e}.\n'
            'Done in {}.\n'
            '-'.format(M, th, sigma**2, sec2str.sec2str(time.time()-start)))

    InfoOut = {'H': H, 'd': dout, 'PCA_th': th, 'sigma': sigma, 'Ym': Yrrm}

    return (S, InfoOut)


class PcaHandler:
    """Interface to perform PCA.

    The PCA is applied at class initialization based on the input data.
    This same operation can be applied afterward to other data using the
    :code:`direct` and :code:`inverse` methods.

    Attributes
    ----------
    Y: (m, n, l) numpy array
        Multi-band data.
    Y_PCA: (m, n, PCA_th) numpy array
        The data in PCA space.
    mask: optional, (m, n) numpy array
        Spatial sampling mask.
        Default is full sampling.
    PCA_transform: optional, bool
        Flag that sets if PCA should really be applied. This is useful
        in soma cases where PCA has already been applied.
        Default is True.
    verbose: optional, bool
        If True, information is sent to output.
    H: (l, PCA_th) numpy array
        The subspace base.
    Ym: (m, n, l) numpy array
        Matrix whose spectra are all composed of the data spectral mean.
    PCA_th: int
        The estimated data dimension.
    InfoOut: dict
        The dictionary contaning additional information about the reduction.
        See Note.

    Note
    ----
    The InfoOut dictionary containg the thee following keys:

    1. 'H' which is the base of the reduced subspace.
       Its shape is (l, PCA_th) where PCA_th is the estimated data
       dimension.
    2. 'd' which is the evolution of the PCA-eigenvalues after estimation.
    3. 'PCA_th' which is the estimated data dimension.
    4. 'sigma' which is the estimated Gaussian noise standard deviation.
    5. 'Ym' which is a (m, n, l) numpy array where the data mean over bands
       is repeated for each spatial location.

    """

    def __init__(self, Y, mask=None, PCA_transform=True, PCA_th='auto',
                 verbose=True):
        """PcaHandler constructor.

        Arguments
        ----------
        Y: (m, n, l) numpy array
            Multi-band data.
        mask: (m, n) numpy array
            Spatial sampling mask.
        PCA_transform: optional, bool
            Flag that sets if PCA should really be applied. This is useful
            in soma cases where PCA has already been applied.
            Default is True.
        verbose: optional, bool
            If True, information is sent to output.
        """
        _logger.info('Initializing a PcaHandler object.')

        # Test PCA_transform
        if type(PCA_transform) is not bool:
            raise ValueError('The PCA_transform parameter should be boolean.')

        # Save everything
        self.Y = Y
        self.mask = mask
        self.PCA_transform = PCA_transform
        self.PCA_th = PCA_th
        self.verbose = verbose

        # Transform data into PCA
        if self.PCA_transform:
            _logger.info('Performing PCA.')

            Y_PCA, InfoOut = Dimension_Reduction(
                self.Y,
                mask=self.mask,
                PCA_th=self.PCA_th,
                verbose=self.verbose)

            self.H = InfoOut['H']
            self.Ym = InfoOut['Ym']
            self.PCA_th = InfoOut['PCA_th']
            self.InfoOut = InfoOut

        else:
            _logger.info('Not performing PCA.')

            Y_PCA = self.Y.copy()
            self.H = np.eye(self.Y.shape[-1])
            self.Ym = np.zeros(self.Y.shape)
            self.PCA_th = self.Y.shape[-1]
            self.InfoOut = None

        self.Y_PCA = Y_PCA

    def direct(self, X=None):
        """Performs direct PCA transformation.

        The input X array can be data to project into the PCA subspace
        or None. If input is None (which is default), the output will be
        simply self.Y_PCA.

        Caution
        -------
        The input data to transform should have the same shape as the Y
        initial data.

        Arguments
        ---------
        X: (m, n, l) numpy array
            The data to transform into PCA space.

        Returns
        -------
        (m, n, PCA_th) numpy array
            Multi-band data in reduced space.
        """
        if X is None:
            return self.Y_PCA

        else:
            m, n, B = X.shape
            centered_data = (X - self.Ym).reshape((m*n, B)).T
            return (self.H.T @ centered_data).T.reshape(
                (m, n, self.PCA_th))

    def inverse(self, X_PCA):
        """Performs inverse PCA transformation.

        Caution
        -------
        The input data to transform should have the same shape as the
        self.Y_PCA transformed data.

        Arguments
        ---------
        X_PCA: (m, n, PCA_th) numpy array
            The data to transform into data space.

        Returns
        -------
        (m, n, l) numpy array
            Multi-band data after inverse transformation.
        """

        m, n, _ = X_PCA.shape
        M = self.H.shape[0]
        X_tmp = X_PCA.reshape((m*n, self.PCA_th)).T

        return (self.H @ X_tmp).T.reshape((m, n, M)) + self.Ym
