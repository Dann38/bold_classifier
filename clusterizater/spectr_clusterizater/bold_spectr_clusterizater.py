import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.stats import norm

from ..clusterizater import BaseClusterizater
from types_font import BOLD, REGULAR


class BoldSpectralClusterizater(BaseClusterizater):
    def __init__(self, significance_level=0.5):
        self.significance_level = significance_level

    def clusterization(self, x: np.ndarray) -> np.ndarray:
        nearby_x = x.copy()
        nearby_x[:-1] += x[1:]
        nearby_x[1:] += x[:-1]
        nearby_x[0] += x[0]
        nearby_x[-1] += x[-1]
        nearby_x = nearby_x / 3.

        x_vec = [[x[i], nearby_x[i]] for i in range(len(x))]
        spectr = SpectralClustering(n_clusters=2, assign_labels='discretize', random_state=0)
        spectr.fit(x_vec)
        x_clust = spectr.labels_

        x_clust0 = x[x_clust == 0]
        x_clust1 = x[x_clust == 1]

        #  https: // www.tsi.lv / sites / default / files / editor / science / Research_journals / Tr_Tel / 2003 / V1 /
        #  yatskiv_gousarova.pdf
        w1 = np.std(x) * len(x)
        w2 = np.std(x_clust0)*len(x_clust0) + np.std(x_clust1)*len(x_clust1)
        f1 = w2/w1
        p = 2
        n = len(x)
        za1 = norm.ppf(1-self.significance_level, loc=0, scale=1)
        f_cr = 1-2/(np.pi*p) - za1*np.sqrt(abs(2*(1-8/(np.pi**2/p))/(n*p)))

        if f1 > f_cr:
            return np.zeros_like(x)+REGULAR
        if np.mean(x[x_clust == 1]) < np.mean(x[x_clust == 0]):
            x_clust[x_clust == 1] = BOLD
            x_clust[x_clust == 0] = REGULAR
        else:
            x_clust[x_clust == 0] = BOLD
            x_clust[x_clust == 1] = REGULAR

        return x_clust
