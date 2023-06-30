import numpy as np

from ..clusterizater import BaseClusterizater
from sklearn.cluster import SpectralClustering


class BoldSpectralClusterizater(BaseClusterizater):

    def clusterization(self, X: np.ndarray) -> np.ndarray:
        # if np.std(X) < 0.05:
        #     return np.zeros_like(X)
        XX = X.copy()
        XX[:-1] += X[1:]
        XX[1:] += X[:-1]
        XX[0] += X[0]
        XX[-1] += X[-1]
        XX = XX / 3.
        X_vec = [[X[i], XX[i]] for i in range(len(X))]
        spectr = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0)
        spectr.fit(X_vec)
        X_clust = spectr.labels_

        if np.mean(X[X_clust == 1]) > np.mean(X[X_clust == 0]):
            X_clust = 1 - X_clust
        return X_clust
