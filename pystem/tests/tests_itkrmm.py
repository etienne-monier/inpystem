# -*- coding: utf-8 -*-

import unittest

import numpy as np
import numpy.random as npr
import numpy.testing as npt

import scipy.fftpack

from context import pystem


class Test_itkrmm(unittest.TestCase):

    def test_rec_lratom_one_comp(self):

        # Data size
        d = 10
        # Number of patches
        N = 1000

        # Low rank component to be estimated.
        lrc = npr.rand(d)
        lrc /= np.linalg.norm(lrc)
        # This line and the similar ones below avoid situation were
        # estimated low rank component is the opposite of the true one.
        lrc = lrc * np.sign(lrc.sum())

        # Data
        data = lrc[:, np.newaxis].dot(npr.randn(1, N))

        # Estimation in case of full sampling ------
        #
        # Estimation of lrc
        lrc_hat = pystem.restore.ITKrMM.rec_lratom(data=data)
        lrc_hat = lrc_hat * np.sign(lrc_hat.sum())

        # Assert that lrc is lrc_hat
        npt.assert_allclose(lrc, lrc_hat)

        # Estimation in case of partial sampling ------
        #
        # Mask
        pix_ratio = 0.8
        mask = npr.rand(d, N) < pix_ratio

        # Estimation of lrc
        lrc_hat = pystem.restore.ITKrMM.rec_lratom(
            data=data,
            masks=mask)
        lrc_hat = lrc_hat * np.sign(lrc_hat.sum())

        # Assert that lrc is lrc_hat
        npt.assert_allclose(lrc, lrc_hat, rtol=0.05)

    def test_rec_lratom_two_comp(self):

        # Number of components
        L = 2

        # Data size
        d = 10
        # Number of patches
        N = 1000

        # Low rank component to be estimated.
        lrc = npr.randn(d, L)
        lrc /= np.tile(np.linalg.norm(lrc, axis=0)[np.newaxis, :], [d, 1])

        # Data
        data = lrc.dot(npr.randn(L, N))

        # Estimation in case of full sampling with one initial component ------
        #
        # Estimation of lrc
        lrc_hat = pystem.restore.ITKrMM.rec_lratom(data=data, lrc=lrc[:, 0])
        s = np.linalg.svd(
            np.hstack((lrc, lrc_hat[:, np.newaxis])),
            compute_uv=False
            )

        # Asserts last singular value less than 1e-10
        self.assertLess(s[-1], 1e-10)

        # Estimation in case of partial sampling with one initial comp. ------
        #
        # Mask
        pix_ratio = 0.8
        mask = npr.rand(d, N) < pix_ratio

        # Estimation of lrc
        lrc_hat = pystem.restore.ITKrMM.rec_lratom(
            data=data,
            masks=mask,
            lrc=lrc[:, 0])
        s = np.linalg.svd(
            np.hstack((lrc, lrc_hat[:, np.newaxis])),
            compute_uv=False
            )

        # Asserts last singular value less than 1e-10
        self.assertLess(s[-1], 1e-2)

    def tes_OMPm(self):
        """
        """
        # Number of signals.
        N = 1000
        # Signal dimension.
        P = 500
        # Dictionary dimension (should be less that P).
        K = 50
        # Number of atoms needed to describe the data (should be less
        # than K).
        L = 5

        # Dictionary
        D = npr.rand(K, P)
        # Dictionary atoms normalization
        D = D / np.tile(np.linalg.norm(D, axis=1)[:, np.newaxis], [1, P])
        # Sparse code
        A = np.hstack((npr.rand(N, L), np.zeros((N, K-L))))
        for row in range(A.shape[0]):
            npr.shuffle(A[row, :])
        # Signal
        X = A @ D

        # OMPm in case full sampling.o
        A_hat = pystem.restore.ITKrMM.OMPm(D, X, L).toarray()

        # Checks A_hat and A close.
        npt.assert_allclose(A_hat, A)

        # # OMPm in case partial sampling.
        # Masks = npr.rand(N, P) < 0.8
        # A_hat = pystem.restore.ITKrMM.OMPm(D, X, L, Masks=Masks).toarray()

        # # Checks A_hat and A close.
        # npt.assert_allclose(A_hat, A)

    def test_itkrmm(self):
        """Test for itkrmm.

        This test mimics some result experiments from the original paper.

        References
        ----------
        Naumova, Valeriya, and Karin Schnass. "Fast dictionary learning from
        incomplete data." EURASIP journal on advances in signal processing
        2018.1 (2018): 12.
        """

        #
        # Create low rank components and dictionary -----------

        # Data dimension
        d = 256
        # DCT basis
        dct_matrix = scipy.fftpack.dct(np.eye(d), norm='ortho')

        # Number of low rank components
        L = 2
        # Low rank romponents
        lrc = dct_matrix[:L, :].T

        # Number of dico atoms
        K = d - L
        # Dico
        dico = dct_matrix[L:, :].T

        #
        # Create signal -----------
        N = 10000

        # Parameters
        e_g, b_g, S, b_S, gamma, s_m = 1/3, 0.15, 6, 0.1, 1/(4*np.sqrt(d)), 4

        # lrc coefficients drawing.
        a, b = 1-b_g, 1
        c_g = (b - a) * npr.random(N) + a
        sigma_l = np.sign(npr.randn(L, N))
        v = np.tile(c_g, [L, 1]) * sigma_l
        v = e_g * v / np.tile(np.linalg.norm(v, axis=0), [L, 1])

        # Sparse code coeff. drawing
        a = 1-b_S
        c_S = (b - a) * npr.random(N) + a
        sigma_k = np.sign(npr.randn(K, N))
        x = np.tile(c_S, [K, 1]) * sigma_k
        for cnt in range(N):
            x[npr.permutation(K)[:K-S], cnt] = 0
        x = (1 - e_g) * x / np.tile(np.linalg.norm(x, axis=0), [d-L, 1])

        # Gaussian noise vector
        r = gamma * npr.randn(d, N)
        # Scaling factor
        s = s_m * npr.random(N)
        s = np.tile(s, [d, 1])

        # Compute signal.
        data = s * (lrc @ v + dico @ x + r)/np.sqrt(
            1 + np.tile(np.linalg.norm(r, axis=0)**2, [d, 1]))

        # Estimate dico
        Dico_hat, _, _ = pystem.restore.ITKrMM.itkrmm(
            data=data,
            K=K,
            S=S,
            lrc=lrc)
        Dico_hat = Dico_hat[:, L:]

        # Compute error
        # E = np.linalg.norm(dico - np.linalg.pinv(Dico_hat)*dico, 'fro')

        # Seaches if all dico atoms are recovered
        scores = np.amax(np.abs(Dico_hat.T @ dico), axis=0)

        #
        T = 0.9
        np.sum(scores > T)
