from transformers import BertConfig, BertModel

from .model import Model


class MBertQA(Model):
    """mBERT model for Question Answering Tasks"""

    def __init__(
        self, pretrained_model_name: str = "bert-base-multilingual-cased"
    ) -> None:
        config = BertConfig()
        mbert = BertModel.from_pretrained(
            pretrained_model_name, add_pooling_layer=False
        )
        Model.__init__(self, mbert, config)
