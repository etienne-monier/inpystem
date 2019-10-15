import numpy as np
import logging
import time

import scipy.sparse as sps


def improve_atom(data, masks, dico, coeffs, j):
    """This function performs dictionary update for atom #j.

    In case the j'th atom is not used, a new atom is chosen among the data
    and the third output is set to True (False otherwise).

    Arguments
    ---------
    data: (d, N) numpy array
        The (corrupted) training signals as its columns.
    masks: (d, N) numpy array
        The masks as its columns.
    dico: (d, K) numpy array
        Initialisation, unit norm column matrix.
    coeffs: (K, N) numpy array
        The sparse codding.
    j: int
        The atom indice to update.

    Returns
    -------
    (d, ) numpy array
        The updated atom.
    (K, N) numpy array
        The updated atoms
    redrawn: int
        1 if a new atom has been generated, 0 therwise.
    """

    # All data indices i that uses the j'th dictionary element,
    # i.e. s.t. coeffs[j, i] != 0.
    nnz = np.flatnonzero(coeffs[j, :])

    if nnz.size == 0:
        # No data uses the j'th atom.
        # In this case, this atom should be replaced.
        #
        # To replace this atom, the data which is has the greatest
        # reconstruction error should be chosen.

        error = data - dico @ coeffs
        error_norm = np.sum(error**2, axis=0)
        pos = np.argmax(error_norm)

        best_atom = data[:, pos]  # other possibility: error[:,pos]

        # Normalization
        best_atom = np.sign(best_atom[0]) * best_atom / np.linalg.norm(
            best_atom)

        coeffs[j, :] = 0
        redrawn = 1

    else:

        redrawn = 0

        tmp_coeffs = coeffs[:, nnz]

        # The coefficients of the element we now improve are not relevant.
        tmp_coeffs[j, :] = 0

        # Vector of errors that we want to minimize with the new element.
        errors = data[:, nnz] - dico*tmp_coeffs

        #
        # wKSVD update:
        #    min || beta.*(errors - atom*coeff) ||_F^2 for beta = mask
        #
        Nit = 10  # The paper suggests 10-20 but 10 is fine and faster.

        for i in range(Nit):

            NewF = \
                masks[:, nnz]*errors + \
                (np.ones((masks.shape[0], nnz.size)) -
                    masks[:, nnz])*(best_atom.dot(coeffs))

            [best_atom, s, beta] = sps.linalg.svds(sps.csc_matrix(NewF), 1)

            coeff_new = s * beta

        coeffs[j, nnz] = coeff_new

    return best_atom, coeffs, redrawn


def dico_cleanup(data, dico, coeffs):
    """This function replaces all atoms:

    - which have a twin which is too close
    - which is not enough used to represent the data

    with data which have maximal representation error.

    Arguments
    ---------
    data: (d, N) numpy array
        The (corrupted) training signals as its columns.
    dico: (d, K) numpy array
        Initialisation, unit norm column matrix.
    coeffs: (K, N) numpy array
        The sparse codding.

    Returns
    -------
    (d, N) numpy array
        The dictionary after cleanup.
    """

    T2 = 0.99
    T1 = 3

    # # of atoms in dico.
    K = dico.shape[1]

    # The approx. error for all data. That's a (N, ) array.
    Er = sum((data - dico @ coeffs)**2, 0)

    # G[i, j] = | <dico_i, dico_j> if i != j
    #           | 0 otherwise.
    G = dico.T @ dico
    G -= np.diag(np.diag(G))

    for k in range(K):

        # If :
        #   - the atom #k has a twin which is too close
        #   - less than T1 data use the atom #k for their representation code
        #
        # Then the atom #k is replaced by the data which have maximal
        # representation error.
        if np.max(G[k, :]) > T2 or np.flatnonzero(
                np.abs(coeffs[k, :]) > 1e-7).size <= T1:

            # Get the index of the data which have the maximal
            # representation error.
            pos = np.argmax(Er)

            # Its error is put to 0 not to be choosen in the future.
            Er[pos] = 0

            # Update new atom
            dico[:, k] = data[:, pos]/np.linalg.norm(data[:, pos])

            # Update G.
            G = dico.T @ dico
            G -= np.diag(np.diag(G))

    return dico


def wKSVD(
        data, masks=None, K=None, S=1, lrc=None, Nit=50, init=None,
        verbose=True, xref=None, preserve_DC=True):

    # Data and masks
    #
    # d is patch size, N is # of patches.
    d, N = data.shape

    if masks is None:
        masks = np.ones(data.shape)

    data = data*masks  # safeguard

    # Dico init
    #
    if K is None:
        K = data.shape[0]

    if N < K-1:
        logging.warning(
            'Less training signals than atoms: trivial solution is data.')
        return data, None

    if init is not None and not np.array_equal(init.shape, np.array([d, K])):
        logging.warning(
            'Initialisation does not match dictionary shape. ',
            'Random initialisation used.')
        init = None

    if init is None:
        init = np.random.randn(d, K)

    # Initialize dico.
    if lrc is not None:
        dico = np.hstack((lrc, init))
        K = dico.shape[1]
    else:
        dico = init

    # dico is normalized.
    dico /= np.tile(np.linalg.norm(dico, axis=0), [K, 1]).T

    # In case preserve_DC is True, compute it and remove it from
    # initialization.
    if preserve_DC:
        DC_atom = np.ones(d)/np.sqrt(d)
        DC_coeffs = np.dot(DC_atom, dico)
        dico -= DC_atom[:, np.newaxis] @ DC_coeffs[np.newaxis, :]

        # dico is normalized (again as DC component has been removed)
        dico /= np.tile(np.linalg.norm(dico, axis=0), [K, 1]).T

    if xref is None:
        xref = data

    #
    # The K-SVD algorithm starts here.
    #

    for it in range(Nit):

        start = time.time()

        # Sparse approximation using OMPm with fixed sparsity level S.
        #
        if preserve_DC:
            over_dico = np.hstack((dico, DC_atom))
        else:
            over_dico = dico

        coeffs = OMPm(over_dico, data, S, masks)

        # Dictionary update
        #
        redrawn_cnt = 0

        # Choose a random permutation to scan the dico.
        rPerm = np.random.permutation(dico.shape[1])

        for j in rPerm:

            # Update the j'th atom
            best_atom, coeffs, redrawn = improve_atom(
                data,
                masks,
                over_dico,
                j,
                coeffs,
                S)

            # If DC atom is preserved, remove it from best_atom.
            if preserve_DC:

                DC_coeff = np.dot(DC_atom, best_atom)
                best_atom -= DC_coeff * DC_atom
                best_atom /= np.linalg.norm(best_atom)

            # Update atom in dico.
            dico[:, j] = best_atom

            # Increment the counter for redrawn atoms.
            redrawn_cnt = redrawn_cnt + redrawn

        dt = time.time() - start

        # This is to remove atoms :
        #   - which have a twin which is too close
        #   - which is not enough used to represent the data.
        dico = dico_cleanup(dico, coeffs[:-1, :], data)

    dico_hat = np.hstack((DC_atom, dico)) if preserve_DC else dico

    return dico_hat, {'time': dt, 'redrawn_cnt': redrawn_cnt}
