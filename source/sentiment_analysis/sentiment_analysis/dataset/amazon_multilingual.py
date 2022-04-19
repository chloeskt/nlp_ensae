import argparse
import logging

import pandas as pd
from datasets import Dataset, load_dataset, DatasetDict


def to_pandas(dataset: Dataset) -> pd.DataFrame:
    indices = [i for i in range(len(dataset))]
    df = pd.DataFrame(dataset[indices])
    return df


class AmazonMultilingual:
    logger = logging.getLogger(__name__)

    def __init__(self, language: str, output_dir: str) -> None:
        self.language = language
        self.output_dir = output_dir

    @staticmethod
    def _remove_neutral(data: pd.DataFrame) -> pd.DataFrame:
        where = data["stars"] == 3
        df = data[~where]
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def _set_sentiment(data: pd.DataFrame) -> pd.DataFrame:
        where = data["stars"] <= 2
        data.loc[where, "labels"] = 0
        where = data["stars"] >= 4
        data.loc[where, "labels"] = 1
        data["labels"] = data["labels"].astype(int)
        return data

    @staticmethod
    def _remove_columns(data: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "review_id",
            "product_id",
            "reviewer_id",
            "stars",
            "review_title",
            "language",
            "product_category",
        ]
        data = data.drop(columns=cols)
        return data

    @staticmethod
    def _rename_cols(data: pd.DataFrame) -> pd.DataFrame:
        data = data.rename(columns={"review_body": "sentence"})
        print(data)
        return data

    def clean(self) -> DatasetDict:
        self.logger.info(f"Retrieve dataset in {self.language}")
        datasets = load_dataset("amazon_reviews_multi", self.language)
        print(datasets)

        splits = ["train", "validation", "test"]

        for split in splits:
            self.logger.info(f"Cleaning split {split}")
            data = to_pandas(datasets[split])
            data = self._remove_neutral(data)
            data = self._set_sentiment(data)
            data = self._remove_columns(data)
            data = self._rename_cols(data)
            datasets[split] = Dataset.from_pandas(data)
            self.logger.info("Done")

        self._save(datasets)
        print(datasets)
        return datasets

    def _save(self, datasets: DatasetDict) -> None:
        self.logger.info(f"Saving cleaned dataset at {self.output_dir}")
        datasets.save_to_disk(self.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parser for AmazonMultilingual dataset cleaning"
    )

    parser.add_argument(
        "--language",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    amazon_multilingual = AmazonMultilingual(
        language=args.language, output_dir=args.output_dir
    )
    amazon_multilingual.clean()
