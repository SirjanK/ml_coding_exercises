import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from algos.kmeans import KMeansFitter


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
        self._covariances = covariances

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
    
    def get_params(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the parameters of the GMM

        :return: Tuple of weights, means, covariances
        """

        return self._weights, self._means, self._covariances


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

        self._k_means_fitter = KMeansFitter(n_clusters=self._n_clusters)  # for initialization

    def fit(self, data: np.ndarray, num_iter=20) -> List[GMM]:
        """
        Fit the GMM on data - return fitted GMM for each step with the last one
        holding the final fitted GMM.

        :param data: Input data to fit the GMM on
        :param num_iter: Number of iterations for the EM algorithm
        :return: List of GMM objects, one for each iteration
        """

        # initialize the parameters
        weights, means, covariances = self._initialize_parameters(data)

        # iterate through the EM algorithm
        gmm_list = []
        for _ in range(num_iter):
            gmm_list.append(GMM(weights, means, covariances))
            weights, means, covariances = self._compute_next_params(
                data,
                weights, 
                means, 
                covariances)
        
        return gmm_list
    
    def _initialize_parameters(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize the parameters of the GMM

        :param data: Input data to fit the GMM on, shape (n_samples, n_features)
        :return: Tuple of weights, means, covariances
        """

        # Fit KMeans
        cluster_centroids = self._k_means_fitter.fit(data)

        # initialize weights to be uniform and the cov to be identity
        return (
            np.ones(self._n_clusters) / self._n_clusters,
            cluster_centroids,
            np.stack([np.eye(self._n_features) for _ in range(self._n_clusters)]),
        )
    
    def _compute_next_params(self, 
                             data: np.ndarray, 
                             curr_weights: np.ndarray, 
                             curr_means: np.ndarray, 
                             curr_covariances: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute the next set of parameters for the GMM using the EM algorithm

        :param data: observation data
        :param curr_weights: Current weights of the GMM components
        :param curr_means: Current means of the GMM components
        :param curr_covariances: Current covariances of the GMM components
        :return: Tuple of weights, means, covariances
        """

        # compute joint probabilities of the data and latent mixture index given the current distribution parameters
        # normalization factors - shape (n_clusters,), one for each cluster
        normalization_factors = 1 / ((2 * np.pi)**(self._n_features / 2) * np.sqrt(np.linalg.det(curr_covariances)))

        # broadcast subtraction of data minus means
        data_minus_means = data[:, np.newaxis, :] - curr_means[np.newaxis, :, :]  # shape (n_samples, n_clusters, n_features)

        # expand dimensions accordingly to compute the exponent term
        data_minus_means_expanded = data_minus_means[:, :, :, np.newaxis]
        # compute exponent term
        exponent = -0.5 * data_minus_means_expanded.transpose(0, 1, 3, 2) @ np.linalg.inv(curr_covariances[np.newaxis, :, :, :]) @ data_minus_means_expanded
        exponent = exponent.squeeze()  # shape (n_samples, n_clusters)
        joint_data_latent_probabilities = normalization_factors * np.exp(exponent) * curr_weights  # shape (n_samples, n_clusters)

        # marginalize across the latent to get probability of data given the current distribution parameters
        data_probabilities = np.sum(joint_data_latent_probabilities, axis=1)
        # compute data probabilities conditioned on latent
        data_probabilities_conditioned_on_latent = joint_data_latent_probabilities / data_probabilities[:, np.newaxis]

        # compute the new weights
        assignment_sum = np.sum(data_probabilities_conditioned_on_latent, axis=0)
        optimal_weights = assignment_sum / np.sum(assignment_sum)  # shape (n_clusters,)

        # compute the new means
        optimal_means = data_probabilities_conditioned_on_latent.T @ data
        optimal_means = optimal_means / assignment_sum[:, np.newaxis]  # shape (n_clusters, n_features)

        # compute the new covariances
        data_minus_optimal_means = data[:, np.newaxis, :] - optimal_means[np.newaxis, :, :]  # shape (n_samples, n_clusters, n_features)
        outer_product = data_minus_optimal_means[:, :, :, np.newaxis] @ data_minus_optimal_means[:, :, np.newaxis, :]  # shape (n_samples, n_clusters, n_features, n_features)
        weighted_products = data_probabilities_conditioned_on_latent[:, :, np.newaxis, np.newaxis] * outer_product  # shape (n_samples, n_clusters, n_features, n_features)
        optimal_covs = np.sum(weighted_products, axis=0) / assignment_sum[:, np.newaxis, np.newaxis]  # shape (n_clusters, n_features, n_features)

        return optimal_weights, optimal_means, optimal_covs
