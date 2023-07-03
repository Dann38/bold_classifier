from dataset_reader.dataset_reader import Reader
from bold_classifier import PsBoldClassifier, MeanBoldClassifier, HistBoldClassifier
import os
from dataset_reader.drawer import Drawer

list_name_data_set = ["ВКР", "ГОСТ", "Геометрия"]
classifier_list = [PsBoldClassifier, MeanBoldClassifier, HistBoldClassifier]

reader = Reader()

for name_data_set in list_name_data_set:
    path_data = os.path.join(os.getcwd(), "dataset")
    name_data = os.path.join(path_data, name_data_set)
    pages = reader.get_array_pages(name_data)

    print("*"*15, name_data_set, "*"*15)
    for classifier in classifier_list:
        cl = classifier()
        rez = cl.evalusion_on_dataset(pages)
        print("-"*10, classifier.__name__, "-"*10)
        for kei in rez.keys():
            print(f"{kei+':':{12}} {rez[kei]:.2f}")

# drawer = Drawer()
# num_page = 7
# classifier = classifier_list[0]
# rez = classifier.classify(pages[num_page].image, pages[num_page].bboxes)
# print(classifier.clusterizater.evalution_listlist(rez, pages[num_page].style))
# drawer.imshow(pages[num_page].image, pages[num_page].bboxes, rez)
