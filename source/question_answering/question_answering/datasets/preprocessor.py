import datasets


class Preprocessor:
    def __init__(self, datasets: datasets.dataset_dict.DatasetDict) -> None:
        self.datasets = datasets
        pass

    @staticmethod
    def _add_answer_end_position(
        row: datasets.arrow_dataset.Example,
    ) -> datasets.arrow_dataset.Example:
        answer = row["answers"]
        try:
            gold_text = answer["text"][0]
        except IndexError:
            answer["answer_end"] = []
            return row
        try:
            start_idx = answer["answer_start"][0]
        except IndexError:
            answer["answer_end"] = []
            return row
        end_idx = start_idx + len(gold_text)
        answer["answer_end"] = [end_idx]
        return row

    def preprocess(self):
        if (
            "validation" in self.datasets.column_names
            and "train" not in self.datasets.column_names
        ):
            self.datasets["validation"] = self.datasets["validation"].map(
                self._add_answer_end_position
            )
        elif (
            "train" in self.datasets.column_names
            and "validation" in self.datasets.column_names
        ):
            self.datasets["train"] = self.datasets["train"].map(
                self._add_answer_end_position
            )
            self.datasets["validation"] = self.datasets["validation"].map(
                self._add_answer_end_position
            )
        elif (
            "id"
            and "context"
            and "question"
            and "answers" in self.datasets.column_names
        ):
            self.datasets = self.datasets.map(self._add_answer_end_position)
        else:
            raise NotImplementedError
        return self.datasets
