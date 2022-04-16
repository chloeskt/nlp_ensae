import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, OrderedDict, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
from datasets import DatasetDict, Dataset
from transformers import (
    PreTrainedTokenizer,
    IntervalStrategy,
    SchedulerType,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.data.data_collator import InputDataClass
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.trainer_utils import PredictionOutput

from ..utils import remove_answer_end

QA_METRICS = Tuple[float, float]


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
    few_shot_learning: bool


@dataclass
class DataArguments:
    """
    Data arguments needed to initiate a Trainer
    """

    datasets: DatasetDict
    dataset_name: str
    validation_features: Dataset
    batch_size: int
    tokenizer: PreTrainedTokenizer
    n_best_size: int
    max_answer_length: int
    tokenized_datasets: DatasetDict
    squad_v2: bool


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
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            logging_steps=self.trainer_args.save_steps,
            no_cuda=False if self.trainer_args.device == "cuda" else True,
        )

        # Initiate Hugging Face Trainer
        if self.data_args.dataset_name == "xquad":
            train_dataset = self.data_args.tokenized_datasets["validation"]
        else:
            train_dataset = self.data_args.tokenized_datasets["train"]

        if self.trainer_args.few_shot_learning:
            callbacks = None
        else:
            callbacks = [
                EarlyStoppingCallback(
                    early_stopping_patience=self.trainer_args.early_stopping_patience
                )
            ]

        self.trainer = Trainer(
            self.trainer_args.model,
            args,
            train_dataset=train_dataset,
            eval_dataset=self.data_args.tokenized_datasets["validation"],
            data_collator=self.trainer_args.data_collator,
            tokenizer=self.data_args.tokenizer,
            callbacks=callbacks,
        )

    @abstractmethod
    def _postprocess_qa_predictions(
        self,
        data: Dataset,
        features: Dataset,
        raw_predictions: Union[PredictionOutput, QuestionAnsweringModelOutput],
    ) -> OrderedDict:
        raise NotImplementedError

    def train(self) -> None:
        self.logger.info("Start training")
        self.trainer.train()
        self.logger.info("Training done")

    def evaluate(self, mode: str, features: Optional[Dataset] = None) -> QA_METRICS:
        if mode == "val":
            _features = self.data_args.validation_features
        elif mode == "test":
            _features = features
        else:
            raise ValueError(
                "Mode should either be val or test. If val, the model will be evaluated on validation features"
                "defined in the DataArguments. If test, one must provide a Dataset of features in the correct"
                "format."
            )
        self.logger.info("Predicting on eval/test dataset")
        raw_predictions = self.trainer.predict(_features)
        self.data_args.validation_features.set_format(
            type=self.data_args.validation_features.format["type"],
            columns=list(self.data_args.validation_features.features.keys()),
        )
        self.logger.info("Postprocessing QA predictions")
        final_predictions = self._postprocess_qa_predictions(
            self.data_args.datasets["validation"],
            self.data_args.validation_features,
            raw_predictions.predictions,
        )
        self.logger.info("Computing metrics")
        results = self._compute_metrics(
            self.trainer_args.metric,
            self.data_args.datasets["validation"],
            final_predictions,
            self.data_args.squad_v2,
        )
        if self.data_args.squad_v2:
            return results["f1"], results["exact"]
        else:
            return results["f1"], results["exact_match"]

    def save_model(self) -> None:
        self.logger.info(
            f"Saving best trained model at {self.trainer_args.model_save_path}"
        )
        torch.save(self.trainer.model.state_dict(), self.trainer_args.model_save_path)

    @staticmethod
    def _compute_metrics(
        metric: Any,
        data: Dataset,
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
