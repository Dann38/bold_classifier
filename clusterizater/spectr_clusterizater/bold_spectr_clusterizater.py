import numpy as np

from ..clusterizater import BaseClusterizater
from sklearn.cluster import SpectralClustering

from scipy.stats import norm
class BoldSpectralClusterizater(BaseClusterizater):
    def __init__(self, significance_level=0.5):
        self.significance_level = significance_level
    def clusterization(self, X: np.ndarray) -> np.ndarray:
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

        X_clust0 = X[X_clust == 0]
        X_clust1 = X[X_clust == 1]

        # https: // www.tsi.lv / sites / default / files / editor / science / Research_journals / Tr_Tel / 2003 / V1 / yatskiv_gousarova.pdf
        w1 = np.std(X)*len(X)
        w2 = np.std(X_clust0)*len(X_clust0) + np.std(X_clust1)*len(X_clust1)
        F1 = w2/w1
        p = 2
        n = len(X)
        za1 = norm.ppf(1-self.significance_level, loc=0, scale=1)
        F_cr = 1-2/(np.pi*p) - za1*np.sqrt(abs(2*(1-8/(np.pi**2/p))/(n*p)))

        if F1 > F_cr:
            return np.zeros_like(X)
        if np.mean(X[X_clust == 1]) > np.mean(X[X_clust == 0]):
            X_clust = 1 - X_clust
        return X_clust
