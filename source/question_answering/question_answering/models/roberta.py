from transformers import RobertaConfig, RobertaModel

from .model import Model


class RobertaQA(Model):
    """RoBERTa model for Question Answering Tasks"""

    def __init__(self, pretrained_model_name: str = "roberta-base") -> None:
        config = RobertaConfig()
        roberta = RobertaModel.from_pretrained(
            pretrained_model_name, add_pooling_layer=False
        )
        Model.__init__(self, roberta, config)
