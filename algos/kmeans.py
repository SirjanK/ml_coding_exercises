import numpy as np


class KMeansFitter:
    """
    The KMeansFitter class fits a KMeans model to data with the specified configuration.
    """

    def __init__(self, n_clusters: int) -> None:
        """
        Initialize the fitter given the number of clusters.
        """

        self._n_clusters = n_clusters
    
    def fit(self, data: np.ndarray, num_iter=20) -> np.ndarray:
        """
        Fit the KMeans model on the data.
        
        :param data: Input data to fit the KMeans model on, shape (n_samples, n_features)
        :param num_iter: Number of iterations for the KMeans algorithm
        :return: np.ndarray of cluster means of shape (n_clusters, n_features)
        """
        
        # initialize the centroids
        centroids = self._initialize_centroids(data)
        
        # iterate through the KMeans algorithm
        for _ in range(num_iter):
            cluster_assignments = self._compute_next_assignments(data, centroids)
            centroids = self._compute_next_centroids(data, cluster_assignments)
        
        return centroids
    
    def _initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        """
        Initialize the centroids of the KMeans model.
        
        :param data: Input data to fit the KMeans model on, shape (n_samples, n_features)
        :return: np.ndarray of cluster means of shape (n_clusters, n_features)
        """
        
        # randomly select n_clusters data points as initial centroids
        return data[np.random.choice(data.shape[0], self._n_clusters, replace=False)]
    
    def _compute_next_assignments(self, data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Compute cluster assignments for the data given the centroids.

        :param data: Input data to fit the KMeans model on, shape (n_samples, n_features)
        :param centroids: np.ndarray of cluster means of shape (n_clusters, n_features)
        :return np.ndarray of cluster assignments of shape (n_samples,) holding indices
        """
        
        # compute the squared distances between each data point and each centroid
        data_centroid_diff = data[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # shape (n_samples, n_clusters, n_features)
        distances = np.sum(np.square(data_centroid_diff), axis=-1)  # shape (n_samples, n_clusters)

        # assign each data point to the closest centroid
        return np.argmin(distances, axis=1)
    
    def _compute_next_centroids(self, data: np.ndarray, cluster_assignments: np.ndarray) -> np.ndarray:
        """
        Compute the next centroids given the data and cluster assignments.

        :param data: Input data to fit the KMeans model on, shape (n_samples, n_features)
        :param cluster_assignments: np.ndarray of cluster assignments of shape (n_samples,) holding indices
        :return np.ndarray of cluster means of shape (n_clusters, n_features)
        """

        # compute the new centroids
        cluster_centroids = []
        for cluster_idx in range(self._n_clusters):
            cluster_data = data[cluster_assignments == cluster_idx]
            cluster_centroids.append(np.mean(cluster_data, axis=0))
        
        return np.array(cluster_centroids)
