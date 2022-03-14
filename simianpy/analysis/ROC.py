import concurrent.futures
import numpy as np

def ROC(A, B, nsteps=100):
    """ Vectorized implementation of computing ROC curve from spike times.

    Parameters
    ----------
    A, B : ndarray (n_observations, n_samples)
        Spike times
    nsteps : int, default=100
        Number of steps to compute probabilities for ROC curve

    Returns
    -------
    pA, pB : ndarray (n_observations, n_steps)
    """
    criteria = np.linspace(np.min([A.min(axis=0),B.min(axis=0)],axis=0), np.max([A.max(axis=0),B.max(axis=0)],axis=0), nsteps).T
    criteria = np.expand_dims(criteria,0)
    A.sort(axis=0)
    B.sort(axis=0)
    pA = (np.expand_dims(A,-1)>=criteria).sum(axis=0)/A.shape[0]
    pB = (np.expand_dims(B,-1)>=criteria).sum(axis=0)/B.shape[0]
    return pA, pB

def AUROC(A, B, nsteps=100):
    """ Vectorized implementation of computing AUROC from spike times.

    Parameters
    ----------
    A, B: ndarray (n_observations, n_samples)
        Spike times
    nsteps : int, default=100
        Number of steps to compute probabilities for ROC curve

    Returns
    -------
    AUROC : ndarray (n_observations,)
    """
    pA, pB = ROC(A, B, nsteps)
    area = -np.trapz(pB, pA)
    return area

def AUROC_obs(obs, labels, nsteps=100):
    """ Vectorized implementation of computing AUROC from spike times and a list of labels.

    Parameters
    ----------
    obs : ndarray (n_observations, n_samples)
        Spike times
    labels : ndarray (n_observations, )
        Label of each observation
    nsteps : int, default=100
        Number of steps to compute probabilities for ROC curve

    Returns
    -------
    AUROC : ndarray (n_observations,)
    """
    return AUROC(obs[labels=='A'], obs[labels=='B'], nsteps)

def bootstrap_AUROC(obs, labels, nsteps=100, n_iterations=1000):
    """ Vectorized, multi-threaded implementation of bootstrapping AUROC from spike times and a list of labels.

    Parameters
    ----------
    obs : ndarray (n_observations, n_samples)
        Spike times
    labels : ndarray (n_observations, )
        Label of each observation
    nsteps : int, default=100
        Number of steps to compute probabilities for ROC curve
    n_iterations : int, default=1000
        Number of bootstrap iterations to perform

    Returns
    -------
    actual_auroc, null_auroc: ndarray(n_observations), ndarray (n_observations, n_iterations)
    """
    actual_auroc = AUROC_obs(obs, labels, nsteps)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        null_auroc = list(executor.map(lambda _: AUROC_obs(obs, np.random.permutation(labels), nsteps), range(n_iterations)))
    null_auroc = np.array(null_auroc)
    return actual_auroc, null_auroc