import collections
import logging
from typing import Union, OrderedDict

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.trainer_utils import PredictionOutput

from .trainer import CustomTrainer


# noinspection DuplicatedCode
class CharacterBasedModelTrainer(CustomTrainer):
    """Trainer for character-based models e.g. CANINE"""

    logger = logging.getLogger(__name__)

    def _postprocess_qa_predictions(
        self,
        examples: Dataset,
        features: Dataset,
        raw_predictions: Union[PredictionOutput, QuestionAnsweringModelOutput],
    ) -> OrderedDict:
        if type(raw_predictions) == tuple:
            all_start_logits, all_end_logits = raw_predictions
        else:
            all_start_logits = raw_predictions.start_logits.cpu().numpy()
            all_end_logits = raw_predictions.end_logits.cpu().numpy()

        self.logger.info("Mapping each example in data to its corresponding features")
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        self.logger.info("Looping over all the examples")
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_score = None  # Only used if squad_v2 is True.
            valid_answers = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]

                token_type_ids = features[feature_index]["token_type_ids"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(
                    self.data_args.tokenizer.cls_token_id
                )
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[
                    -1 : -self.data_args.n_best_size - 1 : -1
                ].tolist()
                end_indexes = np.argsort(end_logits)[
                    -1 : -self.data_args.n_best_size - 1 : -1
                ].tolist()

                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or
                        # correspond to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(token_type_ids)
                            or end_index >= len(token_type_ids)
                            or token_type_ids[start_index] == 0
                            or token_type_ids[end_index] == 0
                        ):
                            continue

                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1
                            > self.data_args.max_answer_length
                        ):
                            continue

                        text_answer = self.data_args.tokenizer.decode(
                            features[feature_index]["input_ids"][start_index:end_index]
                        )

                        valid_answers.append(
                            {
                                "score": start_logits[start_index]
                                + end_logits[end_index],
                                "text": text_answer,
                            }
                        )

            if len(valid_answers) > 0:
                best_answer = sorted(
                    valid_answers, key=lambda x: x["score"], reverse=True
                )[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to
                # avoid failure.
                best_answer = {"text": "", "score": 0.0}

            if not self.data_args.squad_v2:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = (
                    best_answer["text"] if best_answer["score"] > min_null_score else ""
                )
                predictions[example["id"]] = answer
        self.logger.info("Done extracting best answers for all examples")
        return predictions
