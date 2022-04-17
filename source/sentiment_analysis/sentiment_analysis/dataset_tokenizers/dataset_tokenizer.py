from datasets import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding


class DatasetTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: str,
        truncation: bool,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation

    def tokenize(self, data: Dataset) -> BatchEncoding:
        return self.tokenizer(
            data["sentence"],
            padding=self.padding,
            truncation=self.truncation,
        )
