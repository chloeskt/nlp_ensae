import collections
import logging
from typing import OrderedDict

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers.trainer_utils import PredictionOutput

from .trainer import (
    CustomTrainer,
)


# noinspection DuplicatedCode
class TokenBasedModelTrainer(CustomTrainer):
    """Trainer for token-based models e.g. BERT"""

    logger = logging.getLogger(__name__)

    def _postprocess_qa_predictions(
        self, examples: Dataset, features: Dataset, predictions: PredictionOutput
    ) -> OrderedDict:
        # ADAPTED FROM:
        # https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/utils_qa.py
        if len(predictions) != 2:
            raise ValueError(
                "`predictions` should be a tuple with two elements (start_logits, end_logits)."
            )
        all_start_logits, all_end_logits = predictions

        if len(predictions[0]) != len(features):
            raise ValueError(
                f"Got {len(predictions[0])} predictions and {len(features)} features."
            )

        self.logger.info("Mapping each example in data to its corresponding features")
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        all_predictions = collections.OrderedDict()

        self.logger.info("Looping over all the examples")
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]

            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum
                # context available in the current feature.
                token_is_max_context = features[feature_index].get(
                    "token_is_max_context", None
                )

                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if (
                    min_null_prediction is None
                    or min_null_prediction["score"] > feature_null_score
                ):
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

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
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if (
                            end_index < start_index
                            or end_index - start_index + 1
                            > self.data_args.max_answer_length
                        ):
                            continue
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if (
                            token_is_max_context is not None
                            and not token_is_max_context.get(str(start_index), False)
                        ):
                            continue

                        prelim_predictions.append(
                            {
                                "offsets": (
                                    offset_mapping[start_index][0],
                                    offset_mapping[end_index][1],
                                ),
                                "score": start_logits[start_index]
                                + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )
            if self.data_args.squad_v2:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions.
            predictions = sorted(
                prelim_predictions, key=lambda x: x["score"], reverse=True
            )[: self.data_args.n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if self.data_args.squad_v2 and not any(
                p["offsets"] == (0, 0) for p in predictions
            ):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(predictions) == 0 or (
                len(predictions) == 1 and predictions[0]["text"] == ""
            ):
                predictions.insert(
                    0,
                    {
                        "text": "empty",
                        "start_logit": 0.0,
                        "end_logit": 0.0,
                        "score": 0.0,
                    },
                )

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file,
            # using the LogSumExp trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not self.data_args.squad_v2:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = (
                    null_score
                    - best_non_null_pred["start_logit"]
                    - best_non_null_pred["end_logit"]
                )
                if score_diff > 0.0:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]
        self.logger.info("Done extracting best answers for all examples")
        return all_predictions
