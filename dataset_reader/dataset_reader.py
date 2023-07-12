import os
import pickle
from typing import List
from typing import Tuple

import cv2
import numpy as np

from .bbox import BBox
from .page import Page


class Reader:
    def read_dataset(self, path_dir: str) -> List[Page]:
        files = os.listdir(path_dir)
        pages = [self.read_page(path_dir, name_file) for name_file in files if self.__is_page(path_dir, name_file)]
        return pages

    def __is_page(self, path_dir: str, name: str) -> bool:
        if not name.split(".")[-1] in ["jpg", "png", "jpeg"]:
            return False

        img_path = os.path.join(path_dir, name)
        if not os.path.isfile(img_path):
            return False

        pkl_path = os.path.join(path_dir, name + ".pkl")
        if not os.path.isfile(pkl_path):
            return False

        return True

    def read_page(self, path_dir: str, name: str) -> Page:
        img_path = os.path.join(path_dir, name)
        pkl_path = os.path.join(path_dir, name + ".pkl")

        image = self.__get_image(path_dir, img_path)
        bboxes, style = self.__get_bboxes_and_style(path_dir, pkl_path)
        page = Page(image, bboxes, style, name)
        return page

    def __get_image(self, path_dir: str, name_image_file: str) -> np.ndarray:
        path_image = os.path.join(path_dir, name_image_file)
        with open(path_image, "rb") as f:
            chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        image = cv2.imdecode(chunk_arr, cv2.IMREAD_COLOR)
        return image

    def __get_bboxes_and_style(self, path_dir: str, name_pkl_file: str) -> Tuple[List[BBox], List[float]]:
        path_image = os.path.join(path_dir, name_pkl_file)
        with open(path_image, 'rb') as f:
            (llist_bboxes, llist_style) = pickle.load(f)
        bboxes = [BBox.from_dict(bbox) for line_i in llist_bboxes for bbox in line_i]
        style = [int(element_style) for line_i in llist_style for element_style in line_i]
        return bboxes, style

