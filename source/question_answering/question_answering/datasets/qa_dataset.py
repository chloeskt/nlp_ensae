from datasets import Dataset as HF_Dataset
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, tokenized_datasets: HF_Dataset, type: str = None) -> None:
        self.type = type
        if type:
            self.tokenized_dataset = tokenized_datasets[self.type]
        self.tokenized_dataset = tokenized_datasets
        self.tokenized_dataset.set_format("torch")

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        input_ids = self.tokenized_dataset["input_ids"][index]
        attention_mask = self.tokenized_dataset["attention_mask"][index]
        token_type_ids = self.tokenized_dataset["token_type_ids"][index]
        start_positions = self.tokenized_dataset["start_positions"][index]
        end_positions = self.tokenized_dataset["end_positions"][index]
        return (
            input_ids,
            attention_mask,
            token_type_ids,
            start_positions,
            end_positions,
        )
