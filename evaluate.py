import os
from typing import List, Dict

from dataset_reader.dataset_reader import Reader
from dataset_reader.page import Page

from bold_classifier import *
from bold_classifier.utils import llist2vector
from bold_classifier.bold_classifier import BaseBoldClassifier


path_dir_dataset = os.path.join(os.getcwd(), "dataset")
list_name_dataset = ["ВКР", "ГОСТ", "Геометрия"]
classifier_list = [PsBoldClassifier, MeanBoldClassifier, HistBoldClassifier]


def get_dataset(name_dataset: str, path_dataset: str = path_dir_dataset) -> List[Page]:
    name_data = os.path.join(path_dataset, name_dataset)
    pages = reader.get_array_pages(name_data)
    return pages


def evaluate_on_dataset(classifier: BaseBoldClassifier, pages: List[Page]) -> Dict:
    count_pages = len(pages)
    evaluate_sum = {
        "precession": 0,
        "recall": 0,
        "F1": 0,
        "accuracy": 0,
        "N": 0
    }

    for num_page in range(count_pages):
        llist = classifier.classify(pages[num_page].image, pages[num_page].bboxes)
        evaluate_method = evaluate_llist(llist, pages[num_page].style)

        for key in evaluate_sum.keys():
            if key == "N":
                evaluate_sum[key] += evaluate_method["N"]
            else:
                evaluate_sum[key] += evaluate_method["N"] * evaluate_method[key]

    for key in evaluate_sum.keys():
        if key != "N":
            evaluate_sum[key] = evaluate_sum[key] / evaluate_sum["N"]

    return evaluate_sum


def evaluate_llist(llist: List[List[float]], llist_true: List[List[float]]) -> Dict:
    len_lines = [len(line) for line in llist_true]
    word_indicators = llist2vector(llist, len_lines)
    word_indicators_true = llist2vector(llist_true, len_lines)

    N = len(word_indicators)

    TX = word_indicators[word_indicators == word_indicators_true]
    FX = word_indicators[word_indicators != word_indicators_true]
    TP = sum(TX)
    TN = len(TX) - TP
    FP = sum(FX)
    FN = len(FX) - FP

    if TP+FP == 0:
        p = 1.
    else:
        p = TP / (TP + FP)
    if TP + FN == 0:
        r = 1.
    else:
        r = TP / (TP + FN)

    if p + r == 0:
        f1 = 0.
    else:
        f1 = 2 * (p * r) / (p + r)
    accuracy = (TP + TN) / N

    return {"N": N, "precession": p, "recall": r, "F1": f1, "accuracy": accuracy}


def print_evaluate(evaluate: Dict) -> None:
    print("-" * 10, classifier.__name__, "-" * 10)
    for kei in evaluate.keys():
        print(f"{kei + ':':{12}} {evaluate[kei]:.2f}")


if __name__ == "__main__":
    reader = Reader()
    for name_dataset in list_name_dataset:
        pages = get_dataset(name_dataset)

        print("*" * 15, name_dataset, "*" * 15)
        for classifier in classifier_list:
            evaluate_rez = evaluate_on_dataset(classifier(), pages)
            print_evaluate(evaluate_rez)
