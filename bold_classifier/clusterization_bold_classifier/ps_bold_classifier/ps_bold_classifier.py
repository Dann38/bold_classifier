import numpy as np
import cv2
from ..clusterization_bold_classifier import ClusterizationBoldClassifier


class PsBoldClassifier(ClusterizationBoldClassifier):
    def evaluation_one_bbox_image(self, image: np.ndarray) -> float:
        new_image = self.image_resize(image, height=10)
        base_line_image = self._get_base_line_image(new_image)  # baseline - main font area
        base_line_image_without_sparces = self._get_rid_spaces(base_line_image)  # removing spaces from a string

        p_img = base_line_image[:, :-1] - base_line_image[:, 1:]
        p_img[abs(p_img) > 0] = 1.
        p_img[p_img < 0] = 0.
        p = p_img.mean()

        s = 1 - base_line_image_without_sparces.mean()

        if p > s or s == 0:
            evaluation = 1.
        else:
            evaluation = p/s
        return evaluation

    def image_resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        # https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        # return the resized image
        return resized