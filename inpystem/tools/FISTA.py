# -*- coding: utf-8 -*-
"""
This module implements interfacing tools for the FISTA algorithm.

For further informations about the FISTA algorithm, have a look at [1]_.

.. [1] BECK, Amir et TEBOULLE, Marc. A fast iterative shrinkage-thresholding
   algorithm for linear inverse problems. SIAM journal on imaging sciences,
   2009, vol. 2, no 1, p. 183-202.
"""

import time
import math
import logging

import numpy as np

_logger = logging.getLogger(__name__)


class FISTA:
    """
    Fast Iterative Shrinkage-Thresholding Algorithm implementation.

    Attributes
    ----------
    f: function
        :math:`C^{1,1}` convex function.
    df: function
        derivative function of f.
    L: float
        Lipshitz contant of f.
    g: function
        Non-smooth function.
    pg: function
        g poximal operator.
    shape: tuple
        The data shape.
    Nit: None, int
        Number of iteration.
        If None, the iterations will stop as soon as the functional
        no longer evolve.
        Default is None.
    init: numpy array
        Init point which shape is the same as the data.
        If None, a random initailization is drawn.
        Default is None.
    verbose: bool
        If True, process informations are sent to the output.
        Default is True.
    Nit_max: int
        Maximum number of iterations.
    tau: float
        Descent step.
    E: numpy array
        Functional evolution across iterations.
    lim: float
        Controlls the stop condition in case Nit is None.
        The smallest lim, the more iterations before stopping.
        lim is usually 1e-4.
    """

    def __init__(
            self, f, df, L, g, pg, shape, Nit=None, init=None, verbose=True):
        """Initialization function for FISTA.

        Arguments
        ---------
        f: function
            :math:`C^{1,1}` convex function.
        df: function
            derivative function of f.
        L: float
            Lipshitz contant of f.
        g: function
            Non-smooth function.
        pg: function
            g poximal operator.
        shape: tuple
            The data shape.
        Nit: None, int
            Number of iteration.
            If None, the iterations will stop as soon as the functional
            no longer evolve.
            Default is None.
        init: numpy array
            Init point which shape is the same as the data.
            If None, a random initailization is drawn.
            Default is None.
        verbose: bool
            If True, process informations are sent to the output.
            Default is True.
        """

        # Check input
        if L <= 0:
            raise ValueError('Input L parameter should be strct. positive.')

        if Nit is not None:
            if Nit <= 0:
                raise ValueError('Input number of iteration is non-positive.')
            if Nit > 1e6:
                raise ValueError('Input number of iterations is really high.')

        if init is not None:
            if init.shape != shape:
                raise ValueError(
                    'Input init shape and shape parameter do not match.')
        else:
            np.random.seed(1)
            init = np.random.randn(*shape)

        if not isinstance(verbose, bool):
            raise ValueError('Input verbose parameter is not boolean.')

        # Save attributes for methods
        _logger.info('Setting up new FISTA optimizer.')

        self.f = f
        self.df = df
        self.g = g
        self.pg = pg
        self.L = L
        self.shape = shape

        self.Nit = Nit
        self.init = init
        self.verbose = verbose

        # Parameters

        # Max number of iterations.
        self.Nit_max = 1000 if self.Nit is None else self.Nit
        # Step.
        self.tau = 0.99 / self.L
        # Functional values across iterations.
        self.E = np.zeros(self.Nit_max)
        # Tunes stop condition when Nit is None.
        self.lim = 1e-4

    def StopCritera(self, n):
        """This function computes a critera that informs about the algorithm
        convergence at step n.

        Arguments
        ---------
        n: int
            Current step

        Returns
        -------
        float
            Value of the critera.
        """
        if np.allclose(self.E[n - 2], 0):
            return None
        else:
            return np.abs(self.E[n-1] - self.E[n - 2])/(
                self.E[n - 2] * self.tau)

    def StopTest(self, n):
        """This function choose if iterations should be stopped at step n.
        If Nit is not None, it returns True as long as n is smaller than Nit.
        If Nit is None, it returns True as long as the functional is evolving
        fast.

        Arguments
        ---------
        n: int
            Current step.

        Returns
        -------
        bool
            Should the iterations go on ?
        """
        # Iterations should be continued as long as n is smaller than
        # Nit.
        if self.Nit is not None:
            return n < self.Nit

        # The result depends on n and the critera.
        else:
            if n < 2:
                return True
            if n >= self.Nit_max:
                return False
            else:
                critera = self.StopCritera(n)

                # Iterations should be stopped as we got close enough to 0.
                if critera is None:
                    if self.verbose:
                        print(
                            'Iterations stopped as the functional is allclose'
                            ' to 0.')
                    return False
                else:
                    return critera > self.lim

    def execute(self):
        """Method that executes the FISTA algorithm.

        Returns
        -------
        numpy array
            The optimum of the optimization problem.
        dict
            Extra informations about convergence.

        Note
        ----
        Infos in output dictionary:

        * :code:`E`: Evolution of the functional along the iterations.
        * :code:`time`: Execution time.

        """
        _logger.info('Starting FISTA optimization.')

        start = time.time()

        #
        # Initialization
        #

        X0 = self.init

        n = 0
        theta = 1
        Xm1 = X0
        Xy = X0

        #
        # Iterations
        #
        while self.StopTest(n):

            # Display info
            if self.verbose:
                if n >= 2:

                    critera = self.StopCritera(n)
                    print(
                        'n: {}, f + g: {:.3e}, critera: {:.5f} '
                        '(goal: {:.1e})'.format(
                            n,
                            self.E[n-1],
                            0 if critera is None else critera,
                            self.lim)
                        )
                else:
                    print('n: {}'.format(n))

            # 1st step - Gradient descent
            X = Xy - self.tau * self.df(Xy)

            # 2nd step - Thresholding
            X = self.pg(X)

            # Update
            thetap1 = 0.5 * (1 + math.sqrt(1 + 4 * theta**2))
            Xy = X + ((theta - 1) / thetap1) * (X - Xm1)
            #
            theta = thetap1
            Xm1 = X

            # Compute cost function and increment
            self.E[n] = self.f(X) + self.g(X)
            n = n + 1
            # import ipdb; ipdb.set_trace()

        self.E = self.E[:n]

        # Output info
        InfoOut = {'E': self.E, 'time': time.time() - start}

        _logger.info('FISTA optimization finished.')

        return X, InfoOut
