import random
from dataclasses import dataclass, field
from typing import Optional

import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
from datasets import load_from_disk, DatasetDict, Dataset
from nlpaug import Augmenter
from transformers import HfArgumentParser


@dataclass
class NoisifierArguments:
    """
    Arguments needed to noisify SQuADv2-like datasets.
    """

    path_to_custom_dataset: str = field(
        default=None, metadata={"help": "Path towards custom dataset"}
    )
    dataset_name: str = field(
        default=None,
        metadata={"help": "Name of the dataset. Either sst2 or sentiment140"},
    )
    output_dir: str = field(
        default=None,
        metadata={
            "help": "Output directory, will be used to store noisy dataset in csv format"
        },
    )
    noise_level: float = field(
        default=None, metadata={"help": "Level of noise to apply on the dataset"}
    )
    augmenter_type: str = field(
        default=None,
        metadata={
            "help": "Type of Augmenter to use. Either: KeyboardAug, RandomCharAug, SpellingAug, BackTranslationAug "
            "(de/en) or OcrAug"
        },
    )
    action: str = field(
        default=None,
        metadata={
            "help": "Type of action to apply if RandomCharAug was chosen. Either: swap, substitute, insert or delete."
        },
    )

    def __post_init__(self) -> None:
        if (self.augmenter_type == "RandomCharAug" and self.action is None) or (
            self.action is not None and self.augmenter_type != "RandomCharAug"
        ):
            raise ValueError(
                "If you set `augmenter_type` to RandomCharAug, please choose an `action`."
                "If you've chosen an `action`, you must choose `augmenter_type`==RandomCharAug for it"
                "to work."
            )


class Noisifier:
    def __init__(
        self,
        datasets: DatasetDict,
        dataset_name: str,
        level: float,
        type: str,
        action: Optional[str],
    ) -> None:
        self.datasets = datasets
        self.dataset_name = dataset_name
        self.level = level
        self.type = type
        self.action = action

    def _get_augmenter(self) -> Augmenter:
        if self.type == "KeyboardAug":
            return nac.KeyboardAug()

        elif self.type == "RandomCharAug":
            return nac.RandomCharAug(action=self.action)

        elif self.type == "SpellingAug":
            return naw.SpellingAug()

        elif self.type == "OcrAug":
            return nac.OcrAug()

        elif self.type == "BackTranslationAug":
            return naw.BackTranslationAug(
                from_model_name="facebook/wmt19-en-de",
                to_model_name="facebook/wmt19-de-en",
            )

        else:
            raise NotImplementedError

    def _augment_text(self, row: Dataset) -> Dataset:
        if self.dataset_name == "sst2":
            text = "sentence"
        elif self.dataset_name == "sentiment140":
            text = "text"
        else:
            raise NotImplementedError
        augmenter = self._get_augmenter()
        if random.random() < self.level:
            row[text] = augmenter.augment(row[text])
        return row

    def augment(self):
        if (
            "train" in self.datasets.column_names
            and "validation" in self.datasets.column_names
            and "test" in self.datasets.column_names
        ):
            self.datasets["train"] = self.datasets["train"].map(self._augment_text)
            self.datasets["validation"] = self.datasets["validation"].map(
                self._augment_text
            )
            self.datasets["test"] = self.datasets["test"].map(self._augment_text)
        else:
            raise NotImplementedError
        return self.datasets


if __name__ == "__main__":
    parser = HfArgumentParser(NoisifierArguments)
    args = parser.parse_args_into_dataclasses()[0]
    datasets = load_from_disk(args.path_to_custom_dataset)

    noisifier = Noisifier(
        datasets=datasets,
        dataset_name=args.dataset_name,
        level=args.noise_level,
        type=args.augmenter_type,
        action=args.action,
    )

    new_datasets = noisifier.augment()

    # saving
    print(f"saving noisy dataset dict at {args.output_dir}")
    new_datasets.save_to_disk(args.output_dir)
    print(new_datasets)

    print("Loading noisy dataset dict")
    datasets = datasets.load_from_disk(args.output_dir)
    print(datasets)
