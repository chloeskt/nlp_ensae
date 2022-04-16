import os
import random

import datasets
import numpy as np
import pandas as pd
import torch
from datasets import Dataset


def set_seed(seed: int) -> None:
    """Set all seeds to make results reproducible"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def to_pandas(dataset: Dataset) -> pd.DataFrame:
    indices = [i for i in range(len(dataset))]
    df = pd.DataFrame(dataset[indices])
    return df


def remove_answer_end(
    data: datasets.arrow_dataset.Dataset,
) -> datasets.arrow_dataset.Dataset:
    answers_to_return = []
    for answer in data["answers"]:
        del answer["answer_end"]
        answers_to_return.append(answer)
    data["answers"] = answers_to_return
    return data


def tokenize_context(row: pd.DataFrame, doc_stride: int) -> pd.DataFrame:
    row["length_context"] = len([ord(char) for char in row["context"]])
    row["length_question"] = len([ord(char) for char in row["question"]])
    row["total_length"] = (
        row["length_context"] + 2 * row["length_question"] + 6 + doc_stride
    )  # + 3 bcse of 2*CLS, 4*SEP
    row = row.drop(columns=["length_context", "length_question"])
    return row


def remove_examples_longer_than_threshold(
    dataset: pd.DataFrame, max_length: int, doc_stride: int
) -> pd.DataFrame:
    columns_to_remove = ["length_context", "length_question", "total_length"]

    if (
        "length_context" not in dataset.columns
        and "length_question" not in dataset.columns
    ):
        dataset = dataset.progress_apply(tokenize_context, args=(doc_stride,), axis=1)
        columns_to_remove.remove("length_context")
        columns_to_remove.remove("length_question")

    where = dataset["total_length"] < max_length
    dataset = dataset[where]
    dataset = dataset.drop(columns=columns_to_remove)
    dataset = dataset.reset_index(drop=True)
    return dataset
