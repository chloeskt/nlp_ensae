from datasets import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding


class DatasetTokenizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length

    def tokenize(self, data: Dataset) -> BatchEncoding:
        return self.tokenizer(
            data["sentence"],
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
        )
