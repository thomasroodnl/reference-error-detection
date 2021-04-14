import pandas as pd
import regex as re

from constants import *


def train_val_test_split(data: pd.DataFrame, train_frac: float, val_frac: float):
    """
    Randomly split input data into train, validation and test sets
    :param data: input data
    :param train_frac: relative amount of training data
    :param val_frac: relative amount of validation data
    :return: training data set, validation data set and test data set according to the following distributions:
             len(train) = round(train_frac * len(data))
             len(val) = round(val_frac * len(data))
             len(test) = round((1-train_frac-val_frac) * len(data))
    """
    assert 0.0 <= train_frac <= 1.0 and 0.0 <= train_frac <= 1.0, "Data set fractions should be between 0.0 and 1.0."

    data = data.sample(frac=1, random_state=RANDOM_SEED)
    train = data[0: int(train_frac*len(data))]
    val = data[int(train_frac*len(data)): int((train_frac+val_frac)*len(data))]
    test = data[int((train_frac+val_frac)*len(data)):]
    return train, val, test


def missing_punctuation_cnn_dailymail(source_text: str):
    """
    Regular expression for adding missing trailing periods in source texts from the CNN/Dailymail dataset
    :param source_text: source text string
    :return: source text string with added periods
    :rtype: str
    """
    return re.sub(r"([^\.'’`\n])\n", r"\1.\n", source_text)


def missing_punctuation_sum(summary: list):
    """
    Simple function that adds missing trailing periods to summaries.
    :param summary: input summary
    :return: summar with added periods
    """
    for i in range(len(summary)):
        if summary[i][-1] != ".":
            summary[i] = summary[i]+"."
    return summary
