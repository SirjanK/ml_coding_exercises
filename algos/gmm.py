import numpy as np
from typing import List
from dataclasses import dataclass


@dataclass
class GMMParams:
    """
    Dataclass to hold the parameters of a GMM model.
    """

    # shape (n_clusters,)
    weights: np.ndarray
    # shape (n_clusters, n_features)
    means: np.ndarray
    # shape (n_clusters, n_features, n_features)
    covariances: np.ndarray


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

    def fit(self, data: np.ndarray, num_iter=20) -> List[GMMParams]:
        """
        Fit the GMM on data - return fitted parameters for each step with the last one
        holding the final parameters.

        :param data: Input data to fit the GMM on
        :param num_iter: Number of iterations for the EM algorithm
        :return: List of GMMParams objects, one for each iteration
        """

        # TODO implement
        return [
            np.random.rand((self.n_clusters,)),
            np.random.rand((self.n_clusters, self.n_features)),
            np.random.rand((self.n_clusters, self.n_features, self.n_features)),
        ]
