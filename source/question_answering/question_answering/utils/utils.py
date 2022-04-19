import os
import random
from typing import Dict, OrderedDict, Any

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


def compute_metrics(
    metric: Any,
    data: datasets.arrow_dataset.Dataset,
    predictions: OrderedDict,
    squad_v2: bool = True,
) -> Dict[str, float]:
    data = data.map(remove_answer_end, batched=True)
    if squad_v2:
        formatted_predictions = [
            {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
            for k, v in predictions.items()
        ]
    else:
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]

    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in data]

    return metric.compute(predictions=formatted_predictions, references=references)


def postprocess_qa_predictions(
    data: Dict[str, List[int]],
    features: Dict[str, List[int]],
    raw_predictions,
    tokenizer: PreTrainedTokenizer,
    n_best_size: int = 5,
    max_answer_length: int = 30,
    squad_v2: bool = True,
):
    if type(raw_predictions) == tuple:
        all_start_logits, all_end_logits = raw_predictions
    else:
        all_start_logits = raw_predictions.start_logits.cpu().numpy()
        all_end_logits = raw_predictions.end_logits.cpu().numpy()

    if type(features) != datasets.arrow_dataset.Dataset:
        features = datasets.Dataset.from_dict(features)
    if type(data) != datasets.arrow_dataset.Dataset:
        data = datasets.Dataset.from_dict(data)

    # map each example in data to its correponding features
    example_id_to_index = {k: i for i, k in enumerate(data["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Let's loop over all the examples!
    for example_index, example in enumerate(data):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None  # Only used if squad_v2 is True.
        valid_answers = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]

            token_type_ids = features[feature_index]["token_type_ids"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(
                tokenizer.cls_token_id
            )
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[
                -1 : -n_best_size - 1 : -1
            ].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(token_type_ids)
                        or end_index >= len(token_type_ids)
                        or token_type_ids[start_index] == 0
                        or token_type_ids[end_index] == 0
                    ):
                        continue

                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    text_answer = tokenizer.decode(
                        features[feature_index]["input_ids"][start_index:end_index]
                    )

                    valid_answers.append(
                        {
                            "score": start_logits[start_index] + end_logits[end_index],
                            "text": text_answer,
                        }
                    )

        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[
                0
            ]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        if not squad_v2:
            predictions[example["id"]] = best_answer["text"]
        else:
            answer = (
                best_answer["text"] if best_answer["score"] > min_null_score else ""
            )
            predictions[example["id"]] = answer

    return predictions, data
