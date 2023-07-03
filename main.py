from dataset_reader.dataset_reader import Reader
from bold_classifier import PsBoldClassifier, MeanBoldClassifier, HistBoldClassifier
import os
from dataset_reader.drawer import Drawer

list_name_data_set = ["ВКР", "ГОСТ", "Геометрия"]
classifier_list = [PsBoldClassifier] #  MeanBoldClassifier, HistBoldClassifierS

reader = Reader()


# ТЕСТ =============================================================
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


# # ПОСМОТЕРТЬ РЕЗУЛЬТАТ ОДНОГО ===============================================
# path_data = os.path.join(os.getcwd(), "dataset")
# name_data = os.path.join(path_data, list_name_data_set[1])
# pages = reader.get_array_pages(name_data)
# drawer = Drawer()
# num_page = 7
# classifier = classifier_list[0]()
# rez = classifier.classify(pages[num_page].image, pages[num_page].bboxes)
# print(classifier.clusterizater.evalution_listlist(rez, pages[num_page].style))
# drawer.imshow(pages[num_page].image, pages[num_page].bboxes, rez)

# СОХРАНИТЬ РЕЗУЛЬТАТ РАБОТЫ
# os.mkdir("rez")
#
# path_data = os.path.join(os.getcwd(), "dataset")
# name_data = os.path.join(path_data, list_name_data_set[2])
# pages = reader.get_array_pages(name_data)
# drawer = Drawer()
# for i in range(len(pages)):
#     num_page = i
#     classifier = classifier_list[0]()
#     rez = classifier.classify(pages[num_page].image, pages[num_page].bboxes)
#     print(classifier.clusterizater.evalution_listlist(rez, pages[num_page].style))
#     drawer.imsave(pages[num_page].image, pages[num_page].bboxes, rez, os.path.join("rez", f"{i}.jpeg"))