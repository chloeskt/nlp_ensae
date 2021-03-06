from typing import Any

import torch
import torch.nn as nn
from transformers import PretrainedConfig

HuggingFaceModelT = Any


class Model(nn.Module):
    """Generic model for Question Answering Tasks"""

    def __init__(self, model: HuggingFaceModelT, config: PretrainedConfig):
        nn.Module.__init__(self)
        self.model = model
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        output_attentions: torch.Tensor = None,
        output_hidden_states: torch.Tensor = None,
        return_dict: bool = None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return self.qa_outputs(outputs[0])
