from typing import List
from .bbox import BBox
import numpy as np


class Page:
    def __init__(self, image: np.ndarray, bboxes: List[List[BBox]],
                 style: List[List[int]]):
        self.image = image
        self.bboxes = bboxes
        self.style = style
