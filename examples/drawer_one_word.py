import os
import cv2

import binarizer
from bold_classifier import PsBoldClassifier, MeanBoldClassifier, MedianBoldClassifier
from dataset_reader.dataset_reader import Reader
from dataset_reader.drawer import Drawer

path_data = os.path.join(os.getcwd(),  os.path.pardir, "dataset")
name_data = os.path.join(path_data, "Геометрия")
num_page = 3
num_word = 10

reader = Reader()
pages = reader.read_dataset(name_data)
drawer = Drawer()
classifier = PsBoldClassifier()
word = pages[num_page].bboxes[num_word]
rez = classifier.classify(pages[num_page].image, pages[num_page].bboxes)
img = drawer.mark_out(pages[num_page].image, [], [])
img = classifier.binarizer.binarize(img)
img = cv2.cvtColor(img*255, cv2.COLOR_GRAY2BGR)
# img = classifier.image_resize(img[word.y_top_left-1:word.y_bottom_right+1, word.x_top_left-1:word.x_bottom_right+1], )
cv2.imshow("img", img)
cv2.waitKey(0)