import abc

from stanza.server import CoreNLPClient


class DetectionResult:
    """Class representing the output of a Detector"""
    def __init__(self, src_sent_indices, sum_sent_indices, messages):
        """
        Initialize detection result
        :param src_sent_indices: list of src sentence indices corresponding to the summary errors
        :param sum_sent_indices: list of summary sentence indices where an error was found
        :param messages: corresponding error messages
        """
        self.src_sent_indices = src_sent_indices
        self.sum_sent_indices = sum_sent_indices
        self.messages = messages

    def error_detected(self):
        """True if one or more errors were detected, False otherwise"""
        return len(self.src_sent_indices) > 0

    def __str__(self):
        return "\n\n".join(self.messages)


class Detector(metaclass=abc.ABCMeta):
    """Abstract class to be inherited by error detectors"""
    def __init__(self, client: CoreNLPClient, identifier: str):
        """
        Initialize Detector
        :param client: CoreNLP Client for annotation
        :param identifier: unique string identifier of the detector
        """
        self.client = client
        self.identifier = identifier

    @abc.abstractmethod
    def run_detection(self, summary: list, source_text: str, log_level=0) -> DetectionResult:
        """
        Run the Detector's error detection
        :param summary: output summary
        :type summary: list (sentence split)
        :param source_text: source text
        :type source_text: str
        :param log_level: level of logging
            0: OFF (display nothing)
            1: INFO (display information in detected errors)
            2: DEBUG (display processing information relevant for debugging)

        :type log_level: int
        :return: result of the error detection
        :rtype: DetectionResult
        """
        pass
