from transformers import BertConfig, BertModel

from .model import Model


class BertQA(Model):
    """Bert model for Question Answering Tasks"""

    def __init__(self, pretrained_model_name: str = "bert-base-uncased") -> None:
        config = BertConfig()
        bert = BertModel.from_pretrained(pretrained_model_name, add_pooling_layer=False)
        Model.__init__(self, bert, config)
