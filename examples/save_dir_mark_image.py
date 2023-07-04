import os
from dataset_reader.dataset_reader import Reader
from dataset_reader.drawer import Drawer
from bold_classifier import PsBoldClassifier

path_dir_rez = "rez_ВКР"
name_dataset = "ВКР"
classifier_class = PsBoldClassifier
os.mkdir(path_dir_rez)

path_data = os.path.join(os.getcwd(), "../dataset")
name_data = os.path.join(path_data, name_dataset)

reader = Reader()

pages = reader.get_array_pages(name_data)
drawer = Drawer()
for i in range(len(pages)):
    num_page = i
    classifier = classifier_class()
    rez = classifier.classify(pages[num_page].image, pages[num_page].bboxes)
    drawer.imsave(pages[num_page].image, pages[num_page].bboxes, rez, os.path.join(path_dir_rez, f"{i}.jpeg"))
