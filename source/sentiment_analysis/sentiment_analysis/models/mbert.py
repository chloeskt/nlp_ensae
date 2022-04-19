from transformers import BertConfig, BertModel

from .model import Model


class MBertQA(Model):
    """mBERT model for Sentiment Classification Tasks"""

    def __init__(
        self, pretrained_model_name: str = "bert-base-multilingual-cased"
    ) -> None:
        config = BertConfig(num_labels=2)
        mbert = BertModel.from_pretrained(pretrained_model_name, config=config)
        Model.__init__(self, mbert, config)
