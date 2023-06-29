from dataset_reader.dataset_reader import Reader
from bold_classifier import PsBoldClassifier, MeanBoldClassifier
from bold_classifier.clasterization_bold_classifier.utils import listlist2vector
import os
path_data = os.path.join(os.getcwd(), "dataset")
geom_data = os.path.join(path_data, "Геометрия")

def evalution(X, X_true):
    N = len(X)

    TX = X[X == X_true]
    FX = X[X != X_true]
    TP = sum(TX)
    TN = len(TX) - TP
    FP = sum(FX)
    FN = len(FX) - FP

    p = TP/(TP+FP)
    r = TP/(TP+FN)
    f1 = 2*(p*r)/(p+r)
    accuracy = (TP+TN)/N

    return {"precession": p, "recall": r, "F1": f1, "accuracy": accuracy}


reader = Reader()
pages = reader.get_array_pages(geom_data)

ps_classifier = PsBoldClassifier()
mean_classifier = MeanBoldClassifier()


rez_ps = ps_classifier.classify(pages[0].image, pages[0].bboxes)
rez_true = pages[0].style

len_lines = [len(line) for line in rez_true]
X = listlist2vector(rez_ps, len_lines)
X_true = listlist2vector(rez_true, len_lines)

print(evalution(X, X_true))


