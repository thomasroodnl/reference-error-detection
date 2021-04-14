import numpy as np
import pandas as pd
from rouge import Rouge

from .data_loading import get_source_text, get_data_by_annotations, get_data_by_index


def find_source_sent(sentence, source_text, rouge_metric="rouge-2"):
    """
    Finds the source sentence corresponding best to a summary sentence based on a ROUGE metric
    :param sentence: summary sentence
    :param source_text: source text
    :param rouge_metric: ROUGE metric to use (default: rouge-2)
    """
    source_sents = source_text.split("\n\n")
    rouge_scores = np.zeros(len(source_sents))
    rouge = Rouge()
    for i in range(len(source_sents)):
        rouge_scores[i] = rouge.get_scores(sentence, source_sents[i])[0][rouge_metric]["f"]*len(source_sents[i])

    return np.argmax(rouge_scores), source_sents[np.argmax(rouge_scores)]


def inspect_index(data: pd.DataFrame, index: int):
    """
    Inspect the summary and source text of an entry by index
    :param data: data set containing the entry
    :param index: index of the entry to be inspected
    """
    entry = get_data_by_index(data, index)
    print("==========================Entry==========================")
    print("Index ->", index)
    inspect_entry(entry)
    print("==========================End-ry==========================")


def inspect_annotations(data: pd.DataFrame, annotation_types: list):
    """
    Inspect all summaries and corresponding source texts that contain a set of annotations
    :param data: data set to inspect
    :param annotation_types: list of annotation categories (str) to inspect
    """
    data = get_data_by_annotations(data, annotation_types)
    for index, entry in data.iterrows():
        print("==========================Entry==========================")
        print("Index ->", index)
        inspect_entry(entry)
        print("==========================End-ry==========================")
        input("Press enter to continue..")


def inspect_entry(entry):
    """
    Inspect a single entry from a data set
    :param entry: entry to inspect
    """
    collected_annotations = []
    for annotator in entry["annotations"].keys():
        for annotation in entry["annotations"][annotator]:
            collected_annotations.append(annotation)
    if len(collected_annotations) > 0:
        print("==========================Summary==========================")

        for collected_annotation in collected_annotations:
            print("Annotation ->", collected_annotation, "\n")
        for sentence in entry["sentences"]:
            print(sentence)
        print("\n\n========================Source text========================")
        print(get_source_text(entry["story_path"]))


def inspect_annotations_aligned(data: pd.DataFrame, annotation_types: list, print_source_text: bool):
    """
    Inspect all summaries and corresponding source texts that contain a set of annotations,
    will show the corresponding source sentence next to each summary sentence.
    :param data: data set to inspect
    :param annotation_types: list of annotation categories (str) to inspect
    :param print_source_text: Whether to print the full source text for each annotation
    :return:
    """
    data = get_data_by_annotations(data, annotation_types)
    for index, entry in data.iterrows():
        collected_annotations = []
        for annotator in entry["annotations"].keys():
            for annotation in entry["annotations"][annotator]:
                collected_annotations.append(annotation)
        if len(collected_annotations) > 0:
            print("==========================Summary==========================")
            for collected_annotation in collected_annotations:
                print("Annotation ->", collected_annotation, "\n")
            for sentence in entry["sentences"]:
                sent_num, source_sent = find_source_sent(sentence, get_source_text(entry["story_path"]), rouge_metric='rouge-1')
                print(f"Summary sentence: {sentence}")
                print(f"Text sentence: {source_sent}\n")
            if print_source_text:
                print("\n\n========================Source text========================")
                print(get_source_text(entry["story_path"]))
