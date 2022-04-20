import argparse
import logging
import os

import datasets as data
import torch
from datasets import Dataset, load_dataset, load_metric, load_from_disk
from tqdm import tqdm

tqdm.pandas()

from transformers import (
    CanineForQuestionAnswering,
    CanineTokenizer,
    IntervalStrategy,
    RobertaTokenizerFast,
    SchedulerType,
    default_data_collator,
    BertTokenizerFast,
    BertForQuestionAnswering,
    RobertaForQuestionAnswering,
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    XLMRobertaTokenizerFast,
)

from question_answering import (
    DataArguments,
    DatasetCharacterBasedTokenizer,
    DatasetTokenBasedTokenizer,
    Preprocessor,
    TrainerArguments,
    CharacterBasedModelTrainer,
    TokenBasedModelTrainer,
    remove_examples_longer_than_threshold,
    set_seed,
    to_pandas,
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

SQUAD_V2_DATASET_NAME = "squad_v2"
SQUAD_DATASET_NAME = "squad"
XQUAD_DATASET_NAME = "xquad"
NOISY_DATASET_NAME = "noisy"
DYNABENCH_DATASET_NAME = "dynabench/qa"
CUAD_DATASET_NAME = "cuad"

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
    max_length: int,
    doc_stride: int,
    n_best_size: int,
    max_answer_length: int,
    squad_v2: bool,
    eval_only: bool,
    path_to_finetuned_model: str,
    dir_data_noisy: str,
    xquad_subdataset_name: str,
    few_shot_learning: bool,
    path_to_custom_cuad: str,
    path_to_custom_dynabench: str,
) -> None:
    logger.info(f"Loading dataset {dataset_name}")
    if dataset_name == XQUAD_DATASET_NAME:
        datasets = load_dataset(dataset_name, xquad_subdataset_name)
    elif dataset_name == NOISY_DATASET_NAME:
        datasets = load_from_disk(dir_data_noisy)
    elif dataset_name == DYNABENCH_DATASET_NAME:
        datasets = data.load_from_disk(path_to_custom_dynabench)
    elif dataset_name == CUAD_DATASET_NAME:
        datasets = data.load_from_disk(path_to_custom_cuad)
    else:
        datasets = load_dataset(dataset_name)

    logger.info("Adding end_answers to each question")
    preprocessor = Preprocessor(datasets)
    datasets = preprocessor.preprocess()

    logger.info(f"Preparing for model {model_name}")
    if model_name in [CANINE_C_MODEL, CANINE_S_MODEL]:
        if "train" in datasets.column_names:
            df_train = to_pandas(datasets["train"])

        df_validation = to_pandas(datasets["validation"])

        logger.info(f"Removing examples longer than threshold for model {model_name}")
        if "train" in datasets.column_names:
            df_train = remove_examples_longer_than_threshold(
                df_train, max_length=max_length * 2, doc_stride=doc_stride
            )
        df_validation = remove_examples_longer_than_threshold(
            df_validation, max_length=max_length * 2, doc_stride=doc_stride
        )
        logger.info("Done removing examples longer than threshold")

        if "train" in datasets.column_names:
            datasets["train"] = Dataset.from_pandas(df_train)
        datasets["validation"] = Dataset.from_pandas(df_validation)

        pretrained_model_name = f"google/{model_name}"
        tokenizer = CanineTokenizer.from_pretrained(pretrained_model_name)
        model = CanineForQuestionAnswering.from_pretrained(pretrained_model_name)

        tokenizer_dataset_train = DatasetCharacterBasedTokenizer(
            tokenizer,
            max_length,
            doc_stride,
            train=True,
            squad_v2=squad_v2,
            language="en",
        )
        tokenizer_dataset_val = DatasetCharacterBasedTokenizer(
            tokenizer,
            max_length,
            doc_stride,
            train=False,
            squad_v2=squad_v2,
            language="en",
        )
    else:
        if model_name == BERT_MODEL:
            pretrained_model_name = "bert-base-uncased"
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
            model = BertForQuestionAnswering.from_pretrained(pretrained_model_name)

        elif model_name == MBERT_MODEL:
            pretrained_model_name = "bert-base-multilingual-cased"
            tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name)
            model = BertForQuestionAnswering.from_pretrained(pretrained_model_name)

        elif model_name == XLM_ROBERTA_MODEL:
            pretrained_model_name = "xlm-roberta-base"
            tokenizer = XLMRobertaTokenizerFast.from_pretrained(pretrained_model_name)
            model = RobertaForQuestionAnswering.from_pretrained(pretrained_model_name)

        elif model_name == ROBERTA_MODEL:
            pretrained_model_name = "roberta-base"
            tokenizer = RobertaTokenizerFast.from_pretrained(pretrained_model_name)
            model = RobertaForQuestionAnswering.from_pretrained(pretrained_model_name)

        elif model_name == DISTILBERT_MODEL:
            pretrained_model_name = "distilbert-base-uncased"
            tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained_model_name)
            model = DistilBertForQuestionAnswering.from_pretrained(
                pretrained_model_name
            )

        else:
            raise NotImplementedError

        tokenizer_dataset_train = DatasetTokenBasedTokenizer(
            tokenizer, max_length, doc_stride, train=True
        )
        tokenizer_dataset_val = DatasetTokenBasedTokenizer(
            tokenizer, max_length, doc_stride, train=False
        )

    tokenized_datasets = datasets.map(
        tokenizer_dataset_train.tokenize,
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )

    validation_features = datasets["validation"].map(
        tokenizer_dataset_val.tokenize,
        batched=True,
        remove_columns=datasets["validation"].column_names,
    )

    data_collator = default_data_collator
    metric = load_metric("squad_v2" if squad_v2 else "squad")

    if eval_only or few_shot_learning:
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
        few_shot_learning=few_shot_learning,
    )
    data_args = DataArguments(
        datasets=datasets,
        dataset_name=dataset_name,
        validation_features=validation_features,
        batch_size=batch_size,
        tokenizer=tokenizer,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        tokenized_datasets=tokenized_datasets,
        squad_v2=squad_v2,
    )

    if model_name in [CANINE_C_MODEL, CANINE_S_MODEL]:
        trainer = CharacterBasedModelTrainer(trainer_args, data_args, model_name)
    elif model_name in [
        BERT_MODEL,
        MBERT_MODEL,
        XLM_ROBERTA_MODEL,
        ROBERTA_MODEL,
        DISTILBERT_MODEL,
    ]:
        trainer = TokenBasedModelTrainer(trainer_args, data_args, model_name)
    else:
        raise NotImplementedError

    # check if we are in eval mode only or not
    if not eval_only:
        logger.info("START TRAINING")
        trainer.train()

    logger.info("START FINAL EVALUATION")
    f1, exact_match = trainer.evaluate(mode="val")
    logger.info(f"Obtained F1-score: {f1}, Obtained Exact Match: {exact_match}")

    if not eval_only:
        # Save best model
        trainer.save_model()


if __name__ == "__main__":
    debug = False
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("datasets").setLevel(logging.ERROR)
    if debug:
        logger.getChild("question_answering.DatasetCharacterBasedTokenizer").setLevel(
            logging.DEBUG
        )

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
        default="squad_v2",
        choices=[
            SQUAD_V2_DATASET_NAME,
            SQUAD_DATASET_NAME,
            XQUAD_DATASET_NAME,
            NOISY_DATASET_NAME,
            DYNABENCH_DATASET_NAME,
            CUAD_DATASET_NAME,
        ],
        required=True,
        help="Name of the dataset to train/evaluate on",
    )
    parser.add_argument(
        "--xquad_subdataset_name",
        type=str,
        default="xquad.en",
        help="Config name for XQuAD dataset",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for training and evaluation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        required=True,
        help="The maximum length of a feature (question and context)",
    )
    parser.add_argument(
        "--doc_stride",
        type=int,
        required=True,
        help="The authorized overlap between two part of the context when splitting it is needed.",
    )
    parser.add_argument(
        "--n_best_size",
        type=int,
        required=True,
        help="Number of best logits scores to consider",
    )
    parser.add_argument(
        "--max_answer_length",
        type=int,
        required=True,
        help="Maximum length of an answer",
    )
    parser.add_argument("--squad_v2", type=bool, default=False)
    parser.add_argument("--eval_only", type=bool, default=False)
    parser.add_argument(
        "--path_to_finetuned_model",
        type=str,
        default=None,
        help="Path towards a previously finetuned model",
    )
    parser.add_argument(
        "--dir_data_noisy",
        type=str,
        default=None,
        help="Path towards noisy data will be used only if `dataset_name` is set to noisy",
    )
    parser.add_argument(
        "--few_shot_learning",
        type=bool,
        default=False,
        help="Set to True to do few-shot learning",
    )
    parser.add_argument(
        "--path_to_custom_cuad",
        type=str,
        help="Path to custom CUAD dataset, made for few-shot learning",
    )
    parser.add_argument("--path_to_custom_dynabench", type=str, help="")

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
        max_length=args.max_length,
        doc_stride=args.doc_stride,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
        squad_v2=args.squad_v2,
        eval_only=args.eval_only,
        path_to_finetuned_model=args.path_to_finetuned_model,
        dir_data_noisy=args.dir_data_noisy,
        xquad_subdataset_name=args.xquad_subdataset_name,
        few_shot_learning=args.few_shot_learning,
        path_to_custom_cuad=args.path_to_custom_cuad,
        path_to_custom_dynabench=args.path_to_custom_dynabench,
    )
