from bold_classifier.bold_classifier import BOLD, REGULAR
from typing import List
from dataset_reader.bbox import BBox
import cv2
import numpy as np
COLOR_BOLD_ROW = (255, 0, 0)
COLOR_OFFSET_ROW = (0, 0, 255)
COLOR_REGULAR_ROW = (0, 255, 0)
OFFSET_ROW = 2
HEIGHT = 800


class Drawer:
    def imshow(self, img: np.ndarray, bboxes: List[List[BBox]], style: List[List[float]]):

        h = img.shape[0]
        w = img.shape[1]

        img_cope = img.copy()

        coef = w / h
        for i in range(len(bboxes)):
            for j in range(len(bboxes[i])):
                border = 1
                word = bboxes[i][j]
                color = (155, 155, 155)

                style_word = style[i][j]
                if style_word == BOLD:
                    color = COLOR_BOLD_ROW
                elif style_word == OFFSET_ROW:
                    color = COLOR_OFFSET_ROW
                elif style_word == REGULAR:
                    color = COLOR_REGULAR_ROW
                cv2.rectangle(img_cope, (word.x_top_left, word.y_top_left),
                              (word.x_bottom_right, word.y_bottom_right), color, border)
        img = cv2.resize(img_cope, (round(coef * HEIGHT), HEIGHT))
        cv2.imshow("img", img)
        cv2.waitKey(0)