from typing import Union

from transformers import CanineConfig, CanineModel

from .model import Model

CANINE_C = "google/canine-c"
CANINE_S = "google/canine-s"


class CanineSA(Model):
    """CANINE model for Sentiment Classification Tasks"""

    def __init__(self, pretrained_model_name: str = Union[CANINE_C, CANINE_S]) -> None:
        config = CanineConfig(num_labels=2)
        canine = CanineModel.from_pretrained(pretrained_model_name)
        Model.__init__(self, canine, config)
