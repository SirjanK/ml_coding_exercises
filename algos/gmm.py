import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class CachedComputeParams:
    """
    Cached params to expedite the computation of the GMM PDF on multiple data points
    """

    normalization_factor: float
    inv_cov: np.ndarray


class GMM:
    """
    The GMM class holds a Gaussian Mixture Model and supports computing the PDF
    """ 

    def __init__(self, weights: np.ndarray, means: np.ndarray, covariances: np.ndarray) -> None:
        """
        Initialize the GMM with the given parameters

        :param weights: Weights of the GMM components, shape (n_clusters,)
        :param means: Means of the GMM components, shape (n_clusters, n_features)
        :param covariances: Covariances of the GMM components, shape (n_clusters, n_features, n_features)
        """

        self._weights = weights
        self._means = means
        n_features = means.shape[1]
        self._n_clusters = weights.shape[0]

        # pre-cache normalization factors along with the inverse of the covariances
        self._cached_compute_params = [
            CachedComputeParams(
                normalization_factor=1 / ((2 * np.pi)**(n_features / 2) * np.sqrt(np.linalg.det(cov))),
                inv_cov=np.linalg.inv(cov)
            )
            for cov in covariances
        ]
    
    def pdf(self, x: np.ndarray) -> float:
        """
        Compute the probability density function of the GMM at the given point

        :param x: Input data of shape (n_features,)
        :return: float pdf
        """

        total_pdf = 0
        for mixture_idx in range(self._n_clusters):
            weight, mean, cached_params = self._weights[mixture_idx], self._means[mixture_idx], self._cached_compute_params[mixture_idx]

            # compute the exponent term
            exponent = -0.5 * (x - mean).T @ cached_params.inv_cov @ (x - mean)
            
            # compute the pdf for this mixture component
            total_pdf += weight * cached_params.normalization_factor * np.exp(exponent)
        
        return total_pdf


class GMMFitter:
    """
    The GMMFitter class fits a GMM of specified configuration to data.

    We initialize with the number of target clusters and data dimension, then call fit() on input data.
    """

    def __init__(self, n_clusters: int, n_features: int) -> None:
        """
        Initialize the GMMFitter

        :param n_clusters: Number of target clusters
        :parma n_features: Number of features in the data
        """

        self._n_clusters = n_clusters
        self._n_features = n_features

    def fit(self, data: np.ndarray, num_iter=20) -> List[GMM]:
        """
        Fit the GMM on data - return fitted GMM for each step with the last one
        holding the final fitted GMM.

        :param data: Input data to fit the GMM on
        :param num_iter: Number of iterations for the EM algorithm
        :return: List of GMM objects, one for each iteration
        """

        # initialize the parameters
        weights, means, covariances = self._initialize_parameters()

        # iterate through the EM algorithm
        gmm_list = []
        for _ in range(num_iter):
            gmm_list.append(GMM(weights, means, covariances))
            weights, means, covariances = self._compute_next_params(weights, means, covariances)
        
        return gmm_list
    
    def _initialize_parameters(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the parameters of the GMM (naively for now)

        :return: Tuple of weights, means, covariances
        """

        # make sure the covariance matrices are positive definite
        rand_cov = np.random.rand(self._n_clusters, self._n_features, self._n_features)
        rand_cov = rand_cov @ rand_cov.transpose(0, 2, 1)
        return (
            np.random.rand(self._n_clusters,),
            np.random.rand(self._n_clusters, self._n_features),
            rand_cov,
        )
    
    def _compute_next_params(self, curr_weights: np.ndarray, curr_means: np.ndarray, curr_covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the next set of parameters for the GMM using the EM algorithm

        :param curr_weights: Current weights of the GMM components
        :param curr_means: Current means of the GMM components
        :param curr_covariances: Current covariances of the GMM components
        :return: Tuple of weights, means, covariances
        """

        # TODO implement
        return curr_weights, curr_means, curr_covariances
