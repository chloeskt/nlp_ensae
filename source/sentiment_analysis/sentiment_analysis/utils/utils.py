import os
import random
from logging import Logger

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
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
    tokenized_datasets: Dataset,
    output_dir: str,
    model_name: str,
    logger: Logger,
) -> None:
    preds = np.argmax(test_predictions.predictions, axis=1)
    tokenized_datasets["validation"].add_column("predictions", preds)
    final_val_df = to_pandas(tokenized_datasets["validation"])
    final_val_df = final_val_df.drop(
        columns=["input_ids", "token_type_ids", "attention_mask", "idx"]
    )
    save_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    final_val_df.to_csv(save_path, index=False)
    logger.info(f"Saved predictions at {save_path}")
