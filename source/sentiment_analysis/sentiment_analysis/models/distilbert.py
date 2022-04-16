from transformers import DistilBertConfig, DistilBertModel

from .model import Model


class DistilBertQA(Model):
    """DistilBert model for Sentiment Classification Tasks"""

    def __init__(self, pretrained_model_name: str = "distilbert-base-uncased") -> None:
        config = DistilBertConfig(num_labels=2)
        distilbert = DistilBertModel.from_pretrained(
            pretrained_model_name, config=config
        )
        Model.__init__(self, distilbert, config)
