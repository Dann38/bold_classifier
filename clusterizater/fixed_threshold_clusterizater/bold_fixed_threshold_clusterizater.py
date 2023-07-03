import numpy as np

from ..clusterizater import BaseClusterizater


class BoldFixedThresholdClusterizater(BaseClusterizater):

    def clusterization(self, X: np.ndarray) -> np.ndarray:
        k = 0.5
        X_cluster = np.zeros_like(X)
        XX = X.copy()
        XX[:-1] += X[1:]
        XX[1:] += X[:-1]
        XX[0] += X[0]
        XX[-1] += X[-1]
        XX = XX / 3.
        std = np.std(X)
        X_cluster[XX+std < k] = 1.
        X_cluster[X < k] = 1.
        X_cluster[XX-std > k] = 0.
        return X_cluster
