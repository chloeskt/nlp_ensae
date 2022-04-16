from transformers import RobertaConfig, RobertaModel

from .model import Model


class RobertaQA(Model):
    """RoBERTa model for Sentiment Classification Tasks"""

    def __init__(self, pretrained_model_name: str = "roberta-base") -> None:
        config = RobertaConfig(num_labels=2)
        roberta = RobertaModel.from_pretrained(
            pretrained_model_name, config=config
        )
        Model.__init__(self, roberta, config)
