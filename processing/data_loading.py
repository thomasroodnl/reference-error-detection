import json

import pandas as pd
from iteration_utilities import deepflatten

from constants import *


def import_data(path: str = ANNOTATION_PATH):
    """
    Import the annotation data
    :param path: path of the annotation data file
    :return: annotation dataset
    :rtype: pd.DataFrame
    """
    data = []
    with open(path, mode="r", encoding="utf-8") as file:
        annotations = file.readlines()
        for annotation in annotations:
            data.append(json.loads(annotation))

        return pd.DataFrame(data)


def get_source_text(story_path: str):
    """
    Get the source text related to the summary of a source text
    :param story_path: path of the source story
    :return: Source text string
    :rtype: str
    """
    path = SOURCE_TEXT_PATH + story_path[5:]
    with open(path, mode="r", encoding="utf-8") as file:
        return file.read().replace("@highlight", "")


def get_data_by_annotations(data: pd.DataFrame, annotation_types: list):
    """
    Get data points that contain a certain set of annotations
    :param data: Input data set
    :param annotation_types: list of annotation type strings
    :return: Subset of data that have the annotations specified by annotation_types
    """
    flattened_annotations = list(map(lambda x: list(deepflatten(list(x.values()), ignore=str)), data["annotations"]))
    return data[list(map(lambda x: set(annotation_types) <= set(x), flattened_annotations))]


def get_data_by_index(data: pd.DataFrame, index: int):
    """
    Get a data point based on index
    :param data: Data set
    :param index: Index of desired data point
    :return: Data point
    """
    return data.loc[index]
