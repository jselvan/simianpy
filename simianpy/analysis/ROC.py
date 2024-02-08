import concurrent.futures

import numpy as np

def ROC(A, B, n_steps=100):
    """ Vectorized implementation of computing ROC curve from spike rates.

    Parameters
    ----------
    A, B : ndarray (n_observations, ...)
        Spike rates
    n_steps : int, default=100
        Number of steps to compute probabilities for ROC curve

    Returns
    -------
    pA, pB : ndarray (n_observations, n_steps)
    """
    criteria = np.linspace(
        np.min([A.min(axis=0), B.min(axis=0)], axis=0),
        np.max([A.max(axis=0), B.max(axis=0)], axis=0),
        n_steps,
    )
    criteria = np.expand_dims(criteria, 0)
    # criteria is now (1, n_steps, ...)

    A.sort(axis=0)
    B.sort(axis=0)
    # expand A and B to (n_observations, n_steps, ...)
    # and find proportion crossing criteria for each step
    pA = (np.expand_dims(A, 1) >= criteria).sum(axis=0) / A.shape[0]
    pB = (np.expand_dims(B, 1) >= criteria).sum(axis=0) / B.shape[0]
    # shape is now (n_steps, ...)
    return pA, pB

def AUROC(A, B, n_steps=100):
    """ Vectorized implementation of computing AUROC from spike rates.

    Parameters
    ----------
    A, B: ndarray (n_observations, n_samples)
        Spike rates
    n_steps : int, default=100
        Number of steps to compute probabilities for ROC curve

    Returns
    -------
    AUROC : ndarray (n_observations,)
    """
    pA, pB = ROC(A, B, n_steps)
    area = -np.trapz(pA, pB, axis=0)
    return area


def AUROC_obs(obs, labels, A='A', B='B', n_steps=100):
    """ Vectorized implementation of computing AUROC from spike rates and a list of labels.

    Parameters
    ----------
    obs : ndarray (n_observations, n_samples)
        Spike rates
    labels : ndarray (n_observations, )
        Label of each observation
    n_steps : int, default=100
        Number of steps to compute probabilities for ROC curve

    Returns
    -------
    AUROC : ndarray (n_observations,)
    """
    return AUROC(obs[labels == A], obs[labels == B], n_steps)


def bootstrap_AUROC(obs, labels, A='A', B='B', n_steps=100, n_iterations=1000):
    """ Vectorized, multi-threaded implementation of bootstrapping AUROC from spike rates and a list of labels.

    Parameters
    ----------
    obs : ndarray (n_observations, n_samples)
        Spike rates
    labels : ndarray (n_observations, )
        Label of each observation
    n_steps : int, default=100
        Number of steps to compute probabilities for ROC curve
    n_iterations : int, default=1000
        Number of bootstrap iterations to perform

    Returns
    -------
    actual_auroc, null_auroc: ndarray(n_observations), ndarray (n_observations, n_iterations)
    """
    actual_auroc = AUROC_obs(obs, labels, A, B, n_steps)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        null_auroc = list(
            executor.map(
                lambda _: AUROC_obs(obs, np.random.permutation(labels), A, B, n_steps),
                range(n_iterations),
            )
        )
    null_auroc = np.array(null_auroc)
    return actual_auroc, null_auroc