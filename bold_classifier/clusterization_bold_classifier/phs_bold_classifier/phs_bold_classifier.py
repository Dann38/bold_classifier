import numpy as np

from ..clusterization_bold_classifier import ClusterizationBoldClassifier

PERMISSIBLE_H_BBOX = 5

class PhsBoldClassifier(ClusterizationBoldClassifier):
    def evaluation_one_bbox_image(self, image: np.ndarray) -> float:
        base_line_image = self._get_base_line_image(image)  # baseline - main font area
        base_line_image_without_sparces = self._get_rid_spaces(base_line_image)  # removing spaces from a string
        base_line_image.dtype = "bool"
        p_img_bool = np.logical_xor(base_line_image[:-1, :-1],  base_line_image[1:, 1:])
        
        p_img = np.ones_like(p_img_bool)
        p_img[p_img_bool] = 0
        p = p_img.sum()

        s_img = 1 - base_line_image_without_sparces
        s = s_img.sum()

        h = base_line_image.shape[0]
        if p == 0:
            evaluation = 1.
        else:
            evaluation = s/p/h
        return evaluation
    

    def _get_base_line_image(self, image: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        if h < PERMISSIBLE_H_BBOX:
            return image
        
        mean_ = image.mean(1)

        w = image.shape[1]
        if w < h*2:
            not_space = mean_ < 0.95
            return image[not_space, :]
                
        a1 = mean_.min()
        a2 = mean_.max()
        mean_len = len(mean_)
        c_min = mean_len
        h_min = 0
        h_max = len(mean_)-1
        for b1 in range(mean_len//2):
            for b2 in range(mean_len//2, mean_len):
                c1 = ((mean_[:b1] - a2)**2).sum()
                c2 = ((mean_[b1:b2] - a1)**2).sum()
                c3 = ((mean_[b2:] - a2)**2).sum()
                
                c = c1+c2+c3
                if c_min > c:
                    c_min = c
                    h_min = b1
                    h_max = b2
        if h_max-h_min < PERMISSIBLE_H_BBOX:
            return image
        return image[h_min:h_max, :]