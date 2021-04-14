from typing import List

import pandas as pd
from stanza.server import CoreNLPClient

from processing import get_source_text, missing_punctuation_cnn_dailymail, missing_punctuation_sum
from .coref_chain_detector import CorefChainDetector
from .detector import Detector


class ErrorDetection:
    """Umbrella class that allows easy access to separate error detectors"""
    def __init__(self, client: CoreNLPClient):
        """
        Initialize umbrella error detection class
        :param client: CoreNLP Client for annotation
        """
        self.client = client
        self.detectors: List[Detector] = [CorefChainDetector(client)]

    def run_all_detectors(self, data: pd.DataFrame, log_level=0):
        """
        Run all error detection methods on the provided data
        :param data: input data
        :param log_level: level of logging
            0: OFF (display nothing)
            1: INFO (display information in detected errors)
            2: DEBUG (display processing information relevant for debugging)
        :return: detection results
        :rtype: List[dict]
        """
        detection_results = []
        for index, entry in data.iterrows():
            if log_level >= 1:
                print("Index ->", index)
                for annotator in entry["annotations"].keys():
                    for annotation in entry["annotations"][annotator]:
                        if len(annotation) > 0:
                            print("Annotation ->", annotation)

            print(entry["sentences"])
            print(get_source_text(entry["story_path"]))
            summary = missing_punctuation_sum(entry["sentences"])

            source_text = missing_punctuation_cnn_dailymail(get_source_text(entry["story_path"]))

            data_point_results = {"data_index": index}
            for detector in self.detectors:
                data_point_results[detector.identifier] = detector.run_detection(summary, source_text, log_level)

            detection_results.append(data_point_results)

        return detection_results

    def run_all_detectors_entry(self, entry: dict, log_level=0):
        """
        Run all error detection methods on a single entry
        :param entry: input entry
        :param log_level: level of logging
            0: OFF (display nothing)
            1: INFO (display information in detected errors)
            2: DEBUG (display processing information relevant for debugging)
        :return: detection results
        :rtype: dict
        """
        summary = missing_punctuation_sum(entry["sentences"])
        source_text = missing_punctuation_cnn_dailymail(get_source_text(entry["story_path"]))

        data_point_results = {}
        for detector in self.detectors:
            data_point_results[detector.identifier] = detector.run_detection(summary, source_text, log_level)

        return data_point_results
