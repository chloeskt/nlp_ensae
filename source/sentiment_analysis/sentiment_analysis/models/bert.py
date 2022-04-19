from transformers import BertConfig, BertModel

from .model import Model


class BertQA(Model):
    """Bert model for Sentiment Classification Tasks"""

    def __init__(self, pretrained_model_name: str = "bert-base-uncased") -> None:
        config = BertConfig(num_labels=2)
        bert = BertModel.from_pretrained(pretrained_model_name, config=config)
        Model.__init__(self, bert, config)
