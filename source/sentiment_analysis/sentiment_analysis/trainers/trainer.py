import logging
import os
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Callable

import numpy as np
import torch
import torch.nn as nn
from datasets import DatasetDict
from transformers import (
    PreTrainedTokenizer,
    IntervalStrategy,
    SchedulerType,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from transformers.data.data_collator import InputDataClass


@dataclass
class TrainerArguments:
    """
    Arguments needed to initiate a Trainer
    """

    model: nn.Module
    learning_rate: float
    lr_scheduler: SchedulerType
    warmup_ratio: float
    save_strategy: IntervalStrategy
    save_steps: int
    epochs: int
    output_dir: str
    metric: Any
    evaluation_strategy: IntervalStrategy
    weight_decay: float
    data_collator: Callable[[List[InputDataClass]], Dict[str, Any]]
    model_save_path: str
    device: str
    early_stopping_patience: int
    metric_for_best_model: str


@dataclass
class DataArguments:
    """
    Data arguments needed to initiate a Trainer
    """

    datasets: DatasetDict
    dataset_name: str
    dataset_config: str
    batch_size: int
    tokenizer: PreTrainedTokenizer
    tokenized_datasets: DatasetDict


class CustomTrainer(ABC):
    """General Trainer signature"""

    logger = logging.getLogger(__name__)

    def __init__(
        self, trainer_args: TrainerArguments, data_args: DataArguments, model_name: str
    ) -> None:
        self.trainer_args = trainer_args
        self.data_args = data_args
        self.model_name = model_name

        # Define training arguments
        args = TrainingArguments(
            output_dir=os.path.join(
                self.trainer_args.output_dir, self.model_name + "-finetuned"
            ),
            evaluation_strategy=self.trainer_args.evaluation_strategy,
            learning_rate=self.trainer_args.learning_rate,
            weight_decay=self.trainer_args.weight_decay,
            num_train_epochs=self.trainer_args.epochs,
            lr_scheduler_type=self.trainer_args.lr_scheduler,
            warmup_ratio=self.trainer_args.warmup_ratio,
            per_device_train_batch_size=self.data_args.batch_size,
            per_device_eval_batch_size=self.data_args.batch_size,
            save_strategy=self.trainer_args.save_strategy,
            save_steps=self.trainer_args.save_steps,
            push_to_hub=False,
            metric_for_best_model=trainer_args.metric_for_best_model,
            load_best_model_at_end=True,
            logging_steps=self.trainer_args.save_steps,
            no_cuda=False if self.trainer_args.device == "cuda" else True,
        )

        callbacks = [
            EarlyStoppingCallback(
                early_stopping_patience=self.trainer_args.early_stopping_patience
            )
        ]

        self.trainer = Trainer(
            self.trainer_args.model,
            args,
            train_dataset=self.data_args.tokenized_datasets["train"],
            eval_dataset=self.data_args.tokenized_datasets["validation"],
            data_collator=self.trainer_args.data_collator,
            tokenizer=self.data_args.tokenizer,
            callbacks=callbacks,
            compute_metrics=self._compute_metrics,
        )

    def train(self) -> None:
        self.logger.info("Start training")
        self.trainer.train()
        self.logger.info("Training done")

    def evaluate(self) -> None:
        self.logger.info("Start evaluation")
        self.trainer.evaluate()
        self.logger.info("Evaluation done")

    def save_model(self) -> None:
        self.logger.info(
            f"Saving best trained model at {self.trainer_args.model_save_path}"
        )
        torch.save(self.trainer.model.state_dict(), self.trainer_args.model_save_path)

    def _compute_metrics(self, eval_predictions: EvalPrediction) -> Dict[str, float]:
        predictions, labels = eval_predictions
        predictions = np.argmax(predictions, axis=1)
        return self.trainer_args.metric.compute(
            predictions=predictions, references=labels
        )
