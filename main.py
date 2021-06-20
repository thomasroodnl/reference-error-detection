from error_detection import ErrorDetection, DetectionResult
from nlp import CoreNLPWrapper
from processing import *


def error_detection():
    """
    Example error detection on training data
    :return: Result of the error detection
    :rtype: DetectionResult
    """
    train, val, test = train_val_test_split(import_data(), train_frac=0.10, val_frac=0.45)
    data = get_data_by_annotations(train, ["Lack of re-writing"])
    core_nlp = CoreNLPWrapper(annotators=['tokenize, ssplit, truecase, pos, lemma, ner, depparse, openie, coref, parse'])
    with core_nlp.get_instance() as client:
        error_detect = ErrorDetection(client)
        # NOTE: Here we only run on 1 entry, in this case the example text/summary pair used in the paper
        return error_detect.run_all_detectors_entry(data.loc[657], log_level=1)


if __name__ == "__main__":
    error_detection()
    # calc_test_performance()
