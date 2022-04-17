import argparse
import logging
import os

import torch
from datasets import load_dataset, load_metric
from transformers import (
    CanineTokenizer,
    IntervalStrategy,
    RobertaTokenizerFast,
    SchedulerType,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    XLMRobertaTokenizerFast,
    DataCollatorWithPadding,
    CanineForSequenceClassification,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
)

from sentiment_analysis import (
    DatasetTokenizer,
    set_seed,
    TrainerArguments,
    DataArguments,
    CustomTrainer,
    save_predictions_to_pandas_dataframe,
)

SEED = 0
set_seed(SEED)

CANINE_S_MODEL = "canine-s"
CANINE_C_MODEL = "canine-c"
BERT_MODEL = "bert"
MBERT_MODEL = "mbert"
XLM_ROBERTA_MODEL = "xlm_roberta"
ROBERTA_MODEL = "roberta"
DISTILBERT_MODEL = "distilbert"

SST2_DATASET_CONFIG = "sst2"
GLUE_DATASET_NAME = "glue"

NUM_LABELS = 2

logger = logging.getLogger(__name__)


def train_model(
    model_name: str,
    learning_rate: float,
    weight_decay: float,
    type_lr_scheduler: SchedulerType,
    warmup_ratio: float,
    save_strategy: IntervalStrategy,
    save_steps: int,
    num_epochs: int,
    early_stopping_patience: int,
    output_dir: str,
    device: str,
    dataset_name: str,
    batch_size: int,
    truncation: bool,
    eval_only: bool,
    path_to_finetuned_model: str,
    dataset_config: str,
    padding: str,
) -> None:
    logger.info(f"Loading dataset {dataset_name}")
    if dataset_name == GLUE_DATASET_NAME:
        logger.info(f"Chosen configuration is {dataset_config}")
        datasets = load_dataset(dataset_name, dataset_config)
    else:
        raise NotImplementedError

    logger.info(f"Preparing for model {model_name}")
    if model_name in [CANINE_C_MODEL, CANINE_S_MODEL]:
        pretrained_model_name = f"google/{model_name}"
        tokenizer = CanineTokenizer.from_pretrained(pretrained_model_name)
        model = CanineForSequenceClassification.from_pretrained(
            pretrained_model_name, num_labels=NUM_LABELS
        )
    else:
        if model_name == BERT_MODEL:
            pretrained_model_name = "bert-base-uncased"
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
            model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=NUM_LABELS
            )

        elif model_name == MBERT_MODEL:
            pretrained_model_name = "bert-base-multilingual-cased"
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
            model = BertForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=NUM_LABELS
            )

        elif model_name == XLM_ROBERTA_MODEL:
            pretrained_model_name = "xlm-roberta-base"
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_model_name)
            model = RobertaForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=NUM_LABELS
            )

        elif model_name == ROBERTA_MODEL:
            pretrained_model_name = "roberta-base"
            tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)
            model = RobertaForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=NUM_LABELS
            )

        elif model_name == DISTILBERT_MODEL:
            pretrained_model_name = "distilbert-base-uncased"
            tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name)
            model = DistilBertForSequenceClassification.from_pretrained(
                pretrained_model_name, num_labels=NUM_LABELS
            )

        else:
            raise NotImplementedError

    dataset_tokenizer = DatasetTokenizer(
        tokenizer=tokenizer,
        padding=padding,
        truncation=truncation,
    )

    tokenized_datasets = datasets.map(
        dataset_tokenizer.tokenize,
        batched=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer, padding=padding)
    metric = load_metric(dataset_name, dataset_config)

    if eval_only:
        logger.info("Loading own finetuned model")
        model.load_state_dict(torch.load(path_to_finetuned_model, map_location=device))

    trainer_args = TrainerArguments(
        model=model,
        learning_rate=learning_rate,
        lr_scheduler=type_lr_scheduler,
        warmup_ratio=warmup_ratio,
        save_strategy=save_strategy,
        save_steps=save_steps,
        epochs=num_epochs,
        output_dir=output_dir,
        metric=metric,
        evaluation_strategy=save_strategy,
        weight_decay=weight_decay,
        data_collator=data_collator,
        model_save_path=os.path.join(
            output_dir, f"{model_name}-finetuned", f"{model_name}_best_model.pt"
        ),
        device=device,
        early_stopping_patience=early_stopping_patience,
        metric_for_best_model="accuracy",
    )

    data_args = DataArguments(
        datasets=datasets,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        batch_size=batch_size,
        tokenizer=tokenizer,
        tokenized_datasets=tokenized_datasets,
    )

    if model_name in [
        CANINE_C_MODEL,
        CANINE_S_MODEL,
        BERT_MODEL,
        MBERT_MODEL,
        XLM_ROBERTA_MODEL,
        ROBERTA_MODEL,
        DISTILBERT_MODEL,
    ]:
        trainer = CustomTrainer(trainer_args, data_args, model_name)
    else:
        raise NotImplementedError

    # check if we are in eval mode only or not
    if not eval_only:
        logger.info("START TRAINING")
        trainer.train()

    logger.info("START FINAL EVALUATION")
    trainer.evaluate()
    logger.info("Final evaluation done")

    logger.info("GET PREDICTIONS")
    mode = "val"
    test_predictions = trainer.predict(mode=mode)
    logger.info("Predictions done")
    if mode == "val":
        results = trainer.evaluate_predictions(test_predictions)
        logger.info(f"Prediction accuracy {results['accuracy']}")
        save_predictions = True
        if save_predictions:
            save_predictions_to_pandas_dataframe(
                test_predictions,
                tokenized_datasets,
                output_dir,
                model_name,
                logger,
            )

    if not eval_only:
        # Save best model
        trainer.save_model()


if __name__ == "__main__":
    debug = False
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("datasets").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(
        description="Parser for training and data arguments"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
        choices=[
            MBERT_MODEL,
            BERT_MODEL,
            CANINE_S_MODEL,
            CANINE_C_MODEL,
            ROBERTA_MODEL,
            XLM_ROBERTA_MODEL,
            DISTILBERT_MODEL,
        ],
        required=True,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=True,
        help="Chosen learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        required=True,
        help="Chosen weight decay for AdamW optimizer",
    )
    parser.add_argument(
        "--type_lr_scheduler", type=str, required=True, help="Type of LR scheduler"
    )
    parser.add_argument(
        "--warmup_ratio", type=float, required=True, help="Warmup ratio"
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        required=True,
        help="Save strategy",
        choices=["steps", "epochs"],
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        required=True,
        help="Number of steps to perform before saving model",
    )
    parser.add_argument(
        "--num_epochs", type=int, required=True, help="Number of epochs to train for"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        required=True,
        help="Patience for early stopping, validation loss is monitored",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to store the model"
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        help="Device to run the code on, either cpu or cuda",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="glue",
        choices=[GLUE_DATASET_NAME],
        required=True,
        help="Name of the dataset to train/evaluate on",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="sst2",
        help="Config name for GLUE dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for training and evaluation",
    )
    parser.add_argument("--eval_only", type=bool, default=False)
    parser.add_argument(
        "--path_to_finetuned_model",
        type=str,
        default=None,
        help="Path towards a previously finetuned model",
    )
    parser.add_argument(
        "--truncation",
        type=bool,
        required=True,
        help="Whether or not tokenizer should truncate the inputs",
    )
    parser.add_argument("--padding", type=str, help="Padding strategy")

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        type_lr_scheduler=args.type_lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        output_dir=args.output_dir,
        device=args.device,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        truncation=args.truncation,
        eval_only=args.eval_only,
        path_to_finetuned_model=args.path_to_finetuned_model,
        dataset_config=args.dataset_config,
        padding=args.padding,
    )
