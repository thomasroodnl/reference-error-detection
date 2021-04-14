from typing import List

import numpy as np
import stanza
from rouge import Rouge

from .detector import Detector, DetectionResult


class CorefChainDetector(Detector):
    """Error detector that detect reference errors based on coreference chain detection"""
    def __init__(self, client):
        super().__init__(client, "coref_chain_detector")
        self.min_match_score = 0.20  # Minimal score for a VP match to be retained

    def run_detection(self, summary: list, source_text: str, log_level: int = 0):
        """
        Function to detect co-reference 'Lack of re-writing' errors.
        Detection performed through StanfordNLP coreference resolution and constituency parsing

        Example source text:
            "George went to the store. Meanwhile, Michael was driving to work."
        Example (faulty) summary text:
            "George went to the store. Meanwhile, he was driving to work."

        :param summary: output summary
        :param source_text: source text
        :param log_level: level of logging
            0: OFF (display nothing)
            1: INFO (display information in detected errors)
            2: DEBUG (display processing information relevant for debugging)

        :type log_level: int
        :return: True if an error was detected, False otherwise
        :rtype: DetectionResult
        """
        # Run texts through annotation
        src_annotation = self.client.annotate(source_text,
                                              annotators=['tokenize, ssplit, pos, lemma, ner, '
                                                          'depparse, openie, coref, parse'],
                                              properties={"coref.algorithm": "neural"})
        sum_annotation = self.client.annotate(" ".join(summary),
                                              properties={"coref.algorithm": "neural", "truecase.overwriteText":  True})

        # Find matching phrases through the constituency parse trees
        parse_tree_matches = self.match_np_bo_vp(src_annotation, sum_annotation, log_level)

        # Couple co-reference chains from source text to the matches found
        for chain in src_annotation.corefChain:
            for mention in chain.mention:
                mention_for_coref = src_annotation.mentionsForCoref[mention.mentionID]
                for parse_tree_match in parse_tree_matches:
                    if mention_for_coref.sentNum == parse_tree_match.get_src_sent_index() and \
                            mention_for_coref.startIndex == parse_tree_match.get_src_val("NP_start_i"):
                        parse_tree_match.set_src_chain_id(chain.chainID)
                        parse_tree_match.set_src_chain_representative(self.get_representative_string(src_annotation,
                                                                                                     chain))

        # Couple co-reference chains from summary text to the matches found
        for chain in sum_annotation.corefChain:
            for mention in chain.mention:
                mention_for_coref = sum_annotation.mentionsForCoref[mention.mentionID]
                for parse_tree_match in parse_tree_matches:
                    if mention_for_coref.sentNum == parse_tree_match.get_sum_sent_index() and \
                            mention_for_coref.startIndex == parse_tree_match.get_sum_val("NP_start_i"):
                        parse_tree_match.set_sum_chain_id(chain.chainID)
                        parse_tree_match.set_sum_chain_representative(self.get_representative_string(sum_annotation,
                                                                                                     chain))

        # Extract chain ids
        src_chain_ids = list((map(lambda x: x.get_src_chain_id(), parse_tree_matches)))
        sum_chain_ids = list((map(lambda x: x.get_sum_chain_id(), parse_tree_matches)))

        if log_level >= 1:
            for match in parse_tree_matches:
                print(match.get_src_phrase()["NP_string"], match.get_src_chain_id(),
                      match.get_sum_phrase()["NP_string"], match.get_sum_chain_id())

        # Check inconsistencies between co-reference chains
        src_sum_error_indices = self.check_chain_link_oneway(src_chain_ids, sum_chain_ids)
        sum_src_error_indices = self.check_chain_link_oneway(sum_chain_ids, src_chain_ids)

        # Print detected error information (if log_level >= INFO) and return result
        src_sent_indices = []
        sum_sent_indices = []
        messages = []
        for i_error in src_sum_error_indices:
            message = "Summary coreference chain differs where source chain is equal\n" + \
                      f"--- Summary ---\n" + \
                      f" sentence: {summary[parse_tree_matches[i_error].get_sum_sent_index()]}\n" + \
                      f" referent: {parse_tree_matches[i_error].get_sum_val('NP_string')}\n" + \
                      f" (representative: {parse_tree_matches[i_error].get_sum_chain_representative()})\n" + \
                      f"--- Source text ---\n" + \
                      f" sentence: {' '.join(map(lambda t: t.originalText, src_annotation.sentence[parse_tree_matches[i_error].get_src_sent_index()].token))}\n" + \
                      f" referent: {parse_tree_matches[i_error].get_src_val('NP_string')}\n" + \
                      f" (representative: {parse_tree_matches[i_error].get_src_chain_representative()})\n" + \
                      f" VP match rouge score: {parse_tree_matches[i_error].get_match_rouge_score()}\n"

            src_sent_indices.append(parse_tree_matches[i_error].get_src_sent_index())
            sum_sent_indices.append(parse_tree_matches[i_error].get_sum_sent_index())
            messages.append(message)
            if log_level >= 1:
                print("ERROR DETECTED! " + message)

        for i_error in sum_src_error_indices:
            message = "Source coreference chain differs where summary chain is equal\n" + \
                      f"--- Summary ---\n" + \
                      f" sentence: {summary[parse_tree_matches[i_error].get_sum_sent_index()]}\n" + \
                      f" referent: {parse_tree_matches[i_error].get_sum_val('NP_string')}\n" + \
                      f" (representative: {parse_tree_matches[i_error].get_sum_chain_representative()})\n" + \
                      f"--- Source text ---\n" + \
                      f" sentence: {' '.join(map(lambda t: t.originalText, src_annotation.sentence[parse_tree_matches[i_error].get_src_sent_index()].token))}\n" + \
                      f" referent: {parse_tree_matches[i_error].get_src_val('NP_string')}\n" + \
                      f" (representative: {parse_tree_matches[i_error].get_src_chain_representative()})\n" + \
                      f" VP match rouge score: {parse_tree_matches[i_error].get_match_rouge_score()}\n"

            src_sent_indices.append(parse_tree_matches[i_error].get_src_sent_index())
            sum_sent_indices.append(parse_tree_matches[i_error].get_sum_sent_index())
            messages.append(message)
            if log_level >= 1:
                print("ERROR DETECTED! " + message)

        detection_result = DetectionResult(src_sent_indices, sum_sent_indices, messages)

        if not detection_result.error_detected() and log_level >= 1:
            print("No error detected.")

        return detection_result

    @staticmethod
    def get_representative_string(annotation: stanza.server.Document, chain):
        """
        Get the token string of the representative mention of a co-reference chain
        :param annotation: the text annotation the chain originated from
        :param chain: the chain from which to retrieve the representative mention
        :return: representative mention
        :rtype: str
        """
        return annotation.mentionsForCoref[chain.mention[chain.representative].mentionID].headString

    @staticmethod
    def check_chain_link_oneway(chain_ids, other_chain_ids):
        """
        Check whether one list of chain ids is inconsistent with respect to the other
        :param chain_ids: base chain id list
        :param other_chain_ids: chain id list to detect inconsistencies from
        :return: indices that where inconsistent
        :rtype: list (of str)

        NOTE: returned indices depend on list order (first encountered entity is accepted as truth)
        """
        chain_id_map = {}
        error_indices = []
        for i in range(len(chain_ids)):
            if chain_ids[i] is not None and other_chain_ids[i] is not None:
                if not chain_ids[i] in chain_id_map.keys():
                    # Initial chain ID link
                    chain_id_map[chain_ids[i]] = other_chain_ids[i]
                else:
                    # Test if established chain ID link persists
                    if chain_id_map[chain_ids[i]] != other_chain_ids[i]:
                        error_indices.append(i)
        return error_indices

    def match_np_bo_vp(self, src_annotation: stanza.server.Document, sum_annotation: stanza.server.Document, log_level=0):
        """
        Match noun phrases (NP) from source and summary text based on their verb phrases (VP)
        :param src_annotation: annotation of the source text
        :param sum_annotation: annotation of the summary text
        :param log_level: level of logging
            0: OFF (display nothing)
            1: INFO (display information in detected errors)
            2: DEBUG (display processing information relevant for debugging)

        :type log_level: int
        :return: List of matches and their metadata
        :rtype: List[ParseTreeMatch]
        """
        rouge = Rouge()

        sum_phrases = []
        sum_sent_ids = []
        for i_sum in range(len(sum_annotation.sentence)):
            phrase_dicts = self.get_np_vp(sum_annotation.sentence[i_sum], log_level)
            sum_phrases.extend(phrase_dicts)
            sum_sent_ids.extend(len(phrase_dicts) * [i_sum])

        src_phrases = []
        src_sent_ids = []
        for i_src in range(len(src_annotation.sentence)):
            phrase_dicts = self.get_np_vp(src_annotation.sentence[i_src], log_level)
            src_phrases.extend(phrase_dicts)
            src_sent_ids.extend(len(phrase_dicts) * [i_src])

        rouge_scores = np.zeros((len(src_phrases), len(sum_phrases)))
        for i_src_phr in range(len(src_phrases)):
            src_phrase = src_phrases[i_src_phr]
            for i_sum_phr in range(len(sum_phrases)):
                sum_phrase = sum_phrases[i_sum_phr]
                if "VP_string" in src_phrase and "VP_string" in sum_phrase:
                    rouge_scores[i_src_phr, i_sum_phr] = rouge.get_scores(sum_phrase["VP_string"],
                                                                          src_phrase["VP_string"])[0]["rouge-2"]["f"]
                else:
                    rouge_scores[i_src_phr, i_sum_phr] = 0

        best_indices = np.argmax(rouge_scores, axis=0)
        parse_tree_matches = [self.ParseTreeMatch(src_sent_ids[best_indices[i]], sum_sent_ids[i],
                                                  src_phrases[best_indices[i]], sum_phrases[i],
                                                  rouge_scores[best_indices[i], i])
                              for i in range(len(best_indices)) if rouge_scores[best_indices[i], i] >=
                              self.min_match_score]

        return parse_tree_matches

    def get_np_vp(self, sentence: stanza.server.Document.sentence, log_level=0):
        """
        Get the noun phrase (NP) and verb phrase (VP) of a sentence

        :param sentence: sentence from annotation
        :param log_level: level of logging
            0: OFF (display nothing)
            1: INFO (display information in detected errors)
            2: DEBUG (display processing information relevant for debugging)

        :return: Noun phrase and verb phrase data
        :rtype: List[dict]
        dict format:
            key             value
            NP              Noun phrase tree node
            NP_string       Noun phrase string
            NP_start_i      Start (word) index of noun phrase in sentence
            NP_end_i        End index of noun phrase in sentence (exclusive)
            VP_||           ||
        """
        phrase_dicts = self.recursive_np_vp_search(sentence.parseTree.child[0])

        if log_level >= 2:
            print("Phrases")
            for phrase_dict in phrase_dicts:
                print(f"NP {phrase_dict['NP_start_i']}-{phrase_dict['NP_end_i']} {phrase_dict['NP_string']}")
                print(f"VP {phrase_dict['VP_start_i']}-{phrase_dict['VP_end_i']} {phrase_dict['VP_string']}")
                print("VP subphrase:")
            print("None")

        return phrase_dicts

    def recursive_value_search(self, node, value):
        """
        Recursively search for a value in a parseTree
        :param node: parseTree node
        :param value: value to search for
        :return: node with the value, start index of the node in the sentence

        NOTE: Returns the first depth-first occurrence of the value
        """
        return self._recursive_value_search(node, value, 0)

    def _recursive_value_search(self, node, value, start_i):
        """
        Recursively search for a value in a parseTree
        See auxiliary function self.recursive_value_search(node, value)

        :param node: parseTree node
        :param value: value to search for
        :param start_i: current sentence start index
        :return: node with the value, start index of the node in the sentence

        NOTE: Returns the first depth-first occurrence of the value
        """
        if node.value == value:
            return node, start_i
        else:
            if len(node.child) > 0:
                for child in node.child:
                    nodes, n_i = self._recursive_value_search(child, value, start_i)
                    start_i = n_i
                    if node is not None:
                        return node, start_i
                return [], start_i
            else:
                return [], start_i + 1

    def recursive_np_vp_search(self, node):
        """
        Search for noun phrases and verb phrases in a constituency tree
        :param node: root tree node
        :return: list of dictionaries containing the (NP, VP) pairs
        """
        node_pair, np_start_i = self._recursive_np_vp_search(node, 0)
        pair_dicts = []

        while node_pair is not None:
            np_string, np_end_i = self.parse_tree_string(node_pair["NP"], np_start_i)
            pair_dict = {**node_pair, "NP_string": np_string, "NP_start_i": np_start_i, "NP_end_i": np_end_i,
                         "VP_start_i": np_end_i}
            pair_dicts.append(pair_dict)

            node_pair, np_start_i = self._recursive_np_vp_search(pair_dict["VP"], pair_dict["VP_start_i"])

        for i in range(len(pair_dicts)):
            pair_dicts[i]["VP_string"], pair_dicts[i]["VP_end_i"] = self.parse_tree_string(pair_dicts[i]["VP"],
                                                                                           pair_dicts[i]["VP_start_i"])

        return pair_dicts

    def _recursive_np_vp_search(self, node, start_i):
        """
        Helper function that recursively searches a tree and returns an (NP, VP) pair
        :param node: Root node of the tree to start at
        :param start_i: start word index
        :return: Noun phrase and verb phrase, new start index
        """
        if len(node.child) > 0:
            nounp = None
            verbp = None
            for child in node.child:
                if child.value == "NP":
                    nounp = child
                if child.value == "VP":
                    verbp = child

            if nounp is not None and verbp is not None:
                return {"NP": nounp, "VP": verbp}, start_i

            else:
                for child in node.child:
                    node_pair, n_i = self._recursive_np_vp_search(child,  start_i)
                    start_i = n_i
                    if node_pair is not None:
                        return node_pair, start_i
                return None, start_i
        else:
            return None, start_i + 1

    def parse_tree_string(self, node, end_i):
        """
        Generate sentence string based on a parseTree node.
        Additionally, return the end index of the parseTree node in the sentence based on input start index

        :param node: parseTree node
        :param end_i: current end index (before recursive call, start_i = end_i)
        :return: recovered sentence string, end index of the string in the sentence
        :rtype: str, int
        """
        string = ""
        if len(node.child) > 0:
            for child in node.child:
                sub_string, e_i = self.parse_tree_string(child, end_i)
                end_i = e_i
                string += sub_string
            return string, end_i
        else:
            return node.value + " ", end_i + 1

    class ParseTreeMatch:
        """Class to encapsulate data of parse tree matches"""

        def __init__(self, src_sent_index: int, sum_sent_index: int, src_phrase: dict, sum_phrase: dict,
                     match_rouge_score: float):
            """
            Initialize the entry
            :param src_sent_index: Sentence index of the source sentence
            :param sum_sent_index: Sentence index of the summary sentence
            :param src_phrase: Source phrase information
            :type src_phrase: dict (see formatting rtype self.get_np_vp)
            :param sum_phrase: Summary phrase information
            :type sum_phrase: dict (see formatting rtype self.get_np_vp)
            :param match_rouge_score: Rouge score of the VP match
            """
            self.src_sent_index = src_sent_index
            self.sum_sent_index = sum_sent_index
            self.src_phrase = src_phrase
            self.sum_phrase = sum_phrase
            self.match_rouge_score = match_rouge_score
            self.src_chain_id = None
            self.sum_chain_id = None
            self.src_chain_representative = None
            self.sum_chain_representative = None

        def get_src_chain_representative(self):
            return self.src_chain_representative

        def set_src_chain_representative(self, representative):
            self.src_chain_representative = representative

        def get_sum_chain_representative(self):
            return self.sum_chain_representative

        def set_sum_chain_representative(self, representative):
            self.sum_chain_representative = representative

        def get_src_phrase(self):
            return self.src_phrase

        def get_sum_phrase(self):
            return self.sum_phrase

        def get_src_chain_id(self):
            return self.src_chain_id

        def set_src_chain_id(self, chain_id):
            self.src_chain_id = chain_id

        def get_sum_chain_id(self):
            return self.sum_chain_id

        def set_sum_chain_id(self, chain_id):
            self.sum_chain_id = chain_id

        def get_src_sent_index(self):
            return self.src_sent_index

        def get_sum_sent_index(self):
            return self.sum_sent_index

        def get_src_val(self, key):
            return self.src_phrase[key]

        def set_src_val(self, key, val):
            self.src_phrase[key] = val

        def get_sum_val(self, key):
            return self.sum_phrase[key]

        def set_sum_val(self, key, val):
            self.sum_phrase[key] = val

        def get_match_rouge_score(self):
            return self.match_rouge_score
