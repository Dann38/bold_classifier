from dataset_reader.dataset_reader import Reader
from bold_classifier import PsBoldClassifier, MeanBoldClassifier

import os
path_data = os.path.join(os.getcwd(), "dataset")
geom_data = os.path.join(path_data, "Геометрия")

reader = Reader()
pages = reader.get_array_pages(geom_data)

ps_classifier = PsBoldClassifier()
mean_classifier = MeanBoldClassifier()
print(ps_classifier.classify(pages[0].image, pages[0].bboxes))
print(pages[0].style)

