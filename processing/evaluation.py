import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from error_detection import ErrorDetection
from nlp import CoreNLPWrapper
from processing import train_val_test_split, import_data, get_data_by_index


def calc_val_performance():
    """Calculate the performance of the error detection model on the test validation"""
    _, val, _ = train_val_test_split(import_data(), train_frac=0.10, val_frac=0.45)
    annotations = import_data("data/lor_val_annotation.json")
    calc_performance(annotations, val)


def calc_test_performance():
    """Calculate the performance of the error detection model on the test set"""
    _, _, test = train_val_test_split(import_data(), train_frac=0.10, val_frac=0.45)
    annotations = import_data("data/lor_test_annotation.json")
    calc_performance(annotations, test)


def calc_performance(annotations: dict, data: pd.Dataframe):
    """
    Calculate the performance of the error detection model given annotations and data
    :param annotations: dictionary containing the annotations corresponding data indices
    :param data: data, (super)set of all data indices in annotations
    """
    core_nlp = CoreNLPWrapper(annotators=['tokenize, ssplit, truecase, pos, lemma, '
                                          'ner, depparse, openie, coref, parse'])
    with core_nlp.get_instance() as client:
        error_detection = ErrorDetection(client)
        results = []

        annotation = annotations.loc[0]
        for i in range(len(annotation["index"])):
            entry = get_data_by_index(data, annotation["index"][i])
            results.append(error_detection.run_all_detectors_entry(entry, log_level=1)
                           ["coref_chain_detector"].error_detected())

    print("Annotation", annotation["error"])
    print("Results", results)
    print("Precision", precision_score(annotation["error"], results),
          "Recall", recall_score(annotation["error"], results),
          "F1", f1_score(annotation["error"], results))

    print("Accuracy", accuracy_score(annotation["error"], results))
    print("Accuracy baseline", accuracy_score(annotation["error"], len(annotation["error"])*[False]))