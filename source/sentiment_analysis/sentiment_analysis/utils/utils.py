import os
import random
from logging import Logger

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers.trainer_utils import PredictionOutput


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


def save_predictions_to_pandas_dataframe(
    test_predictions: PredictionOutput,
    datasets: Dataset,
    output_dir: str,
    model_name: str,
    mode: str,
    logger: Logger,
) -> None:
    preds = np.argmax(test_predictions.predictions, axis=1)
    if mode == "val":
        df = to_pandas(datasets["validation"])
    elif mode == "test":
        df = to_pandas(datasets["test"])
    else:
        raise NotImplementedError
    df["predictions"] = preds
    save_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    df.to_csv(save_path, index=False)
    logger.info(f"Saved predictions at {save_path}")


def remove_neutral_tweets(dataset: DatasetDict) -> DatasetDict:
    for col in dataset.column_names:
        # Keep only non-neutral tweets
        non_neutral_ids = np.where(np.array(dataset[col]["sentiment"]) != 2)[0]
        dataset[col] = dataset[col].select(non_neutral_ids)
    return dataset
