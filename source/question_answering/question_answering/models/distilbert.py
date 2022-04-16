from transformers import DistilBertConfig, DistilBertModel

from .model import Model


class DistilBertQA(Model):
    """DistilBert model for Question Answering Tasks"""

    def __init__(self, pretrained_model_name: str = "distilbert-base-uncased") -> None:
        config = DistilBertConfig()
        distilbert = DistilBertModel.from_pretrained(
            pretrained_model_name, add_pooling_layer=False
        )
        Model.__init__(self, distilbert, config)
