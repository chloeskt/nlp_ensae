from typing import Union

from transformers import CanineConfig, CanineModel

from .model import Model

CANINE_C = "google/canine-c"
CANINE_S = "google/canine-s"


class CanineQA(Model):
    """CANINE model for Question Answering Tasks"""

    def __init__(self, pretrained_model_name: str = Union[CANINE_C, CANINE_S]) -> None:
        config = CanineConfig()
        canine = CanineModel.from_pretrained(
            pretrained_model_name, add_pooling_layer=False
        )
        Model.__init__(self, canine, config)
