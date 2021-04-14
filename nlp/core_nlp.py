import os
from stanza.server import CoreNLPClient
from constants import CORENLP_PATH


class CoreNLPWrapper:
    """Singleton wrapper for the CoreNLP server"""
    instance = None

    def __init__(self, annotators):
        os.environ['CORENLP_HOME'] = CORENLP_PATH
        self.annotators = annotators

    def get_instance(self):
        if self.instance is None:
            self.instance = CoreNLPClient(annotators=self.annotators)
        return self.instance
