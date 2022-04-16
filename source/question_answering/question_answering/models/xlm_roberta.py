from transformers import RobertaConfig, RobertaModel

from .model import Model


class XlmRobertaQA(Model):
    """XLM-RoBERTa model for Question Answering Tasks"""

    def __init__(self, pretrained_model_name: str = "xlm-roberta-base") -> None:
        config = RobertaConfig()
        xlm_roberta = RobertaModel.from_pretrained(
            pretrained_model_name, add_pooling_layer=False
        )
        Model.__init__(self, xlm_roberta, config)
