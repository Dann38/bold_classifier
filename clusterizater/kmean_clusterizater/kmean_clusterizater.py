import numpy as np

from ..clusterizater import BaseClusterizater
from sklearn.cluster import KMeans


class Bold2MeanClusterizater(BaseClusterizater):

    def clusterization(self, X: np.ndarray) -> np.ndarray:
        XX = X.copy()
        XX[:-1] += X[1:]
        XX[1:] += X[:-1]
        XX[0] += X[0]
        XX[-1] += X[-1]
        XX = XX / 3.
        X_vec = [[X[i], XX[i]] for i in range(len(X))]
        kmeans = KMeans(n_clusters=2, n_init="auto")
        kmeans.fit(X_vec)
        X_clust = kmeans.labels_

        cluster0 = kmeans.cluster_centers_[0][0]
        cluster1 = kmeans.cluster_centers_[1][0]

        bold_cluster = min(cluster0, cluster1)
        regular_cluster = max(cluster0, cluster1)
        if cluster0 == bold_cluster:
            X_clust = 1.0 - X_clust
        return X_clust
