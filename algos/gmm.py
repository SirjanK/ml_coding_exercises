import numpy as np
from typing import List


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

        pass
    
    def pdf(self, x: np.ndarray) -> float:
        """
        Compute the probability density function of the GMM at the given point

        :param x: Input data of shape (n_features,)
        :return: float pdf
        """

        # TODO implement
        return 1


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

        self.n_clusters = n_clusters
        self.n_features = n_features

    def fit(self, data: np.ndarray, num_iter=20) -> List[GMM]:
        """
        Fit the GMM on data - return fitted GMM for each step with the last one
        holding the final fitted GMM.

        :param data: Input data to fit the GMM on
        :param num_iter: Number of iterations for the EM algorithm
        :return: List of GMM objects, one for each iteration
        """

        # TODO implement
        rand_cov = np.random.rand(self.n_clusters, self.n_features, self.n_features)
        rand_cov = rand_cov @ rand_cov.transpose(0, 2, 1)
        return [
            GMM(
                weights=np.random.rand(self.n_clusters,),
                means=np.random.rand(self.n_clusters, self.n_features),
                covariances=rand_cov,
            )
        ] * num_iter
