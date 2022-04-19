from transformers import RobertaConfig, RobertaModel

from .model import Model


class XlmRobertaQA(Model):
    """XLM-RoBERTa model for Sentiment Classification Tasks"""

    def __init__(self, pretrained_model_name: str = "xlm-roberta-base") -> None:
        config = RobertaConfig(num_labels=2)
        xlm_roberta = RobertaModel.from_pretrained(pretrained_model_name, config=config)
        Model.__init__(self, xlm_roberta, config)
