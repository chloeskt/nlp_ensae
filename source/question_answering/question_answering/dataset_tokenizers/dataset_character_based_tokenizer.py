import logging
from typing import Dict, Union, List

from datasets import Dataset
from transformers import PreTrainedTokenizer

CANINE_TOKENIZED_EXAMPLES = Dict[str, Union[List[List[int]], List[int]]]


# noinspection DuplicatedCode
class DatasetCharacterBasedTokenizer:
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        doc_stride: int,
        train: bool,
        squad_v2: bool,
        language: str = "en",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.pad_on_right = tokenizer.padding_side == "right"
        self.train = train
        self.language = language
        self.squad_v2 = squad_v2

    def tokenize(self, data: Dataset) -> CANINE_TOKENIZED_EXAMPLES:
        if self.train:
            return self._tokenize_train_data(data)
        else:
            return self._tokenize_val_data(data)

    def _tokenize_val_data(self, data: Dataset) -> CANINE_TOKENIZED_EXAMPLES:
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        data["question"] = [q.lstrip() for q in data["question"]]

        # Tokenize our data with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            data["question" if self.pad_on_right else "context"],
            data["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            padding="max_length",
        )

        final_input_ids = []
        final_attention_masks = []
        final_token_type_ids = []
        final_example_id = []

        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            # get example id needed for evaluation purposes
            final_example_id.append(data["id"][i])

            final_input_ids.append(input_ids)

            # attention_mask
            final_attention_masks.append(tokenized_examples["attention_mask"][i])

            # token_type_ids
            final_token_type_ids.append(tokenized_examples["token_type_ids"][i])

            # compute question length
            sep_index = input_ids.index(self.tokenizer.sep_token_id)
            question_length = len(input_ids[: sep_index + 1])

            # now if for this example we have an overflow
            # create new example with this overflow
            if len(tokenized_examples["overflowing_tokens"][i]) != 0:
                # get example id needed for evaluation purposes
                final_example_id.append(data["id"][i])

                overflow = tokenized_examples["overflowing_tokens"][i]

                # retrieve question to be added at the beginning overflow_input_ids
                sep_index = input_ids.index(self.tokenizer.sep_token_id)
                tokens_question = input_ids[: sep_index + 1]  # +1 to get the SEP
                overflow_input_ids = tokens_question + overflow

                # truncate to max_length-1,  -1 to be able to add SEP at the end
                overflow_input_ids = overflow_input_ids[: self.max_length - 1]

                # attention mask
                attention_mask = [1] * (len(overflow_input_ids) + 1) + [0] * (
                    self.max_length - len(overflow_input_ids)
                )
                # truncate if needed
                attention_mask = attention_mask[: self.max_length]

                final_attention_masks.append(attention_mask)

                # token_type_ids: 0 if question, 1 if context, 0 if padding
                token_type_ids = (
                    [0] * len(tokens_question)
                    + [1] * (len(overflow) + 1)
                    + [0] * (self.max_length - len(overflow_input_ids))
                )[: self.max_length]
                final_token_type_ids.append(token_type_ids)

                # pad input_ids if necessary
                sep_id = self.tokenizer.sep_token_id
                overflow_input_ids = (
                    overflow_input_ids
                    + [sep_id]
                    + [0] * (self.max_length - len(overflow_input_ids) - 1)
                )[: self.max_length]

                # add in input_ids the context
                final_input_ids.append(overflow_input_ids)

        seen_ids = []
        for id in final_example_id:
            if id not in seen_ids:
                seen_ids.append(id)
            else:
                if id != seen_ids[-1]:
                    print(id)
                    raise

        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_masks,
            "token_type_ids": final_token_type_ids,
            "example_id": final_example_id,
        }

    def _tokenize_train_data(self, data: Dataset) -> CANINE_TOKENIZED_EXAMPLES:
        self.logger.info("Start tokenizing dataset for training")
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        data["question"] = [q.lstrip() for q in data["question"]]

        # Tokenize our data with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            data["question" if self.pad_on_right else "context"],
            data["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            padding="max_length",
        )

        final_input_ids = []
        final_attention_masks = []
        final_token_type_ids = []
        final_start_positions = []
        final_end_positions = []

        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            # We will label impossible answers with the index of the CLS token.
            cls_index = input_ids.index(self.tokenizer.cls_token_id)
            final_input_ids.append(input_ids)
            answers = data["answers"][i]

            try:
                rep = answers["text"][0]
            except IndexError:
                rep = ""

            # attention_mask
            final_attention_masks.append(tokenized_examples["attention_mask"][i])

            # token_type_ids
            final_token_type_ids.append(tokenized_examples["token_type_ids"][i])

            # compute question length
            sep_index = input_ids.index(self.tokenizer.sep_token_id)
            question_length = len(input_ids[: sep_index + 1])

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0 & len(answers["answer_end"]) == 0:
                final_start_positions.append(cls_index)
                final_end_positions.append(cls_index)
            elif answers["answer_end"][0] <= self.max_length - question_length - 1:
                start_position = answers["answer_start"][0] + question_length
                final_start_positions.append(start_position)
                end_position = answers["answer_end"][0] + question_length
                final_end_positions.append(end_position)
            else:
                final_start_positions.append(cls_index)
                final_end_positions.append(cls_index)

            ac = self.tokenizer.decode(
                final_input_ids[-1][final_start_positions[-1] : final_end_positions[-1]]
            )

            is_in_first_part = False
            is_in_second_part = False
            ac_overflow = None
            if rep == ac:
                is_in_first_part = True

            # now if for this example we have an overflow
            # create new example with this overflow
            if len(tokenized_examples["overflowing_tokens"][i]) != 0:

                overflow = tokenized_examples["overflowing_tokens"][i]

                # retrieve question to be added at the beginning overflow_input_ids
                sep_index = input_ids.index(self.tokenizer.sep_token_id)
                tokens_question = input_ids[: sep_index + 1]  # +1 to get the SEP
                overflow_input_ids = tokens_question + overflow

                # truncate to max_length-1,  -1 to be able to add SEP at the end
                overflow_input_ids = overflow_input_ids[: self.max_length - 1]

                # attention mask
                attention_mask = [1] * (len(overflow_input_ids) + 1) + [0] * (
                    self.max_length - len(overflow_input_ids)
                )
                # truncate if needed
                attention_mask = attention_mask[: self.max_length]

                final_attention_masks.append(attention_mask)

                # token_type_ids: 0 if question, 1 if context, 0 if padding
                token_type_ids = (
                    [0] * len(tokens_question)
                    + [1] * (len(overflow) + 1)
                    + [0] * (self.max_length - len(overflow_input_ids))
                )[: self.max_length]
                final_token_type_ids.append(token_type_ids)

                # pad input_ids if necessary
                sep_id = self.tokenizer.sep_token_id
                overflow_input_ids = (
                    overflow_input_ids
                    + [sep_id]
                    + [0] * (self.max_length - len(overflow_input_ids) - 1)
                )[: self.max_length]

                # add in input_ids the context
                final_input_ids.append(overflow_input_ids)

                start_character_in_context = (
                    self.max_length - 1 - len(tokens_question) - self.doc_stride
                )

                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0 & len(answers["answer_end"]) == 0:
                    final_start_positions.append(cls_index)
                    final_end_positions.append(cls_index)
                elif (
                    answers["answer_start"][0] >= start_character_in_context
                    and answers["answer_end"][0]
                    < self.max_length
                    - len(tokens_question)
                    - 1
                    + start_character_in_context
                ):
                    relative_start = (
                        answers["answer_start"][0]
                        - start_character_in_context
                        + len(tokens_question)
                    )

                    relative_end = (
                        answers["answer_end"][0]
                        - start_character_in_context
                        + len(tokens_question)
                    )

                    final_start_positions.append(relative_start)
                    final_end_positions.append(relative_end)
                else:
                    final_start_positions.append(cls_index)
                    final_end_positions.append(cls_index)

                ac_overflow = self.tokenizer.decode(
                    final_input_ids[-1][
                        final_start_positions[-1] : final_end_positions[-1]
                    ]
                )
                if rep == ac_overflow:
                    is_in_second_part = True

            if not is_in_first_part and not is_in_second_part:
                self.logger.debug("\n")
                self.logger.debug("\033[91mERROR WITH PREDICTION OF: \033[0m")
                self.logger.debug("True answer: ")
                self.logger.debug(f">{rep}<")
                self.logger.debug("context", data["context"][i])
                self.logger.debug("question", data["question"][i])
                self.logger.debug(answers)
                self.logger.debug(f"answer computed ac >{ac}<")
                self.logger.debug(f"answer computed ac_overflow >{ac_overflow}<")

                # handle special cases depending on language
                if self.squad_v2:
                    languages = ["vi", "ar", "zh", "es", "hi", "ru"]
                else:
                    languages = ["vi", "en", "ar", "zh", "es", "hi", "ru"]
                if self.language in languages:
                    self.logger.debug("\n")
                    self.logger.debug("\033[91mTRYING TO FIX PREDICTIONS: \033[0m")
                    if len(tokenized_examples["overflowing_tokens"][i]) == 0:
                        # no overflow
                        final_start_positions[-1] -= 1
                        final_end_positions[-1] -= 1

                        ac = self.tokenizer.decode(
                            final_input_ids[-1][
                                final_start_positions[-1] : final_end_positions[-1]
                            ]
                        )
                        self.logger.debug(f"new computed answer: >{ac}<")

                        if ac == rep:
                            pass
                        else:
                            # in arabic somethimes two letters are actually missing
                            if self.language == "ar":
                                final_start_positions[-1] -= 1
                                final_end_positions[-1] -= 1

                                ac = self.tokenizer.decode(
                                    final_input_ids[-1][
                                        final_start_positions[-1] : final_end_positions[
                                            -1
                                        ]
                                    ]
                                )
                    else:
                        # there is an overflow, need to know if the answer
                        # is in the overflow or in the first part or in both
                        if ac and ac_overflow:
                            # it is in both
                            final_start_positions[-2] -= 1
                            final_end_positions[-2] -= 1
                            final_start_positions[-1] -= 1
                            final_end_positions[-1] -= 1

                            ac = self.tokenizer.decode(
                                final_input_ids[-2][
                                    final_start_positions[-2] : final_end_positions[-2]
                                ]
                            )
                            self.logger.debug(
                                f"new computed answer instead of ac : >{ac}<"
                            )
                            ac_overflow = self.tokenizer.decode(
                                final_input_ids[-1][
                                    final_start_positions[-1] : final_end_positions[-1]
                                ]
                            )
                            self.logger.debug(
                                f"new computed answer instead of ac_overflow: >{ac_overflow}<"
                            )

                            if ac == rep and ac_overflow == rep:
                                pass
                            else:
                                # in arabic somethimes two letters are actually missing
                                if self.language == "ar":
                                    final_start_positions[-2] -= 1
                                    final_end_positions[-2] -= 1
                                    final_start_positions[-1] -= 1
                                    final_end_positions[-1] -= 1

                                    ac = self.tokenizer.decode(
                                        final_input_ids[-2][
                                            final_start_positions[
                                                -2
                                            ] : final_end_positions[-2]
                                        ]
                                    )
                                    ac_overflow = self.tokenizer.decode(
                                        final_input_ids[-1][
                                            final_start_positions[
                                                -1
                                            ] : final_end_positions[-1]
                                        ]
                                    )

                        elif ac is not None and (
                            ac_overflow is None or ac_overflow == ""
                        ):
                            # only in the first part:
                            final_start_positions[-2] -= 1
                            final_end_positions[-2] -= 1

                            ac = self.tokenizer.decode(
                                final_input_ids[-2][
                                    final_start_positions[-2] : final_end_positions[-2]
                                ]
                            )
                            self.logger.debug(f"new computed answer: >{ac}<")

                            if ac == rep:
                                pass
                            else:
                                if self.language == "ar":
                                    final_start_positions[-2] -= 1
                                    final_end_positions[-2] -= 1

                                    ac = self.tokenizer.decode(
                                        final_input_ids[-2][
                                            final_start_positions[
                                                -2
                                            ] : final_end_positions[-2]
                                        ]
                                    )

                        elif (ac is None or ac == "") and ac_overflow is not None:
                            # only in the overflow
                            final_start_positions[-1] -= 1
                            final_end_positions[-1] -= 1

                            ac_overflow = self.tokenizer.decode(
                                final_input_ids[-1][
                                    final_start_positions[-1] : final_end_positions[-1]
                                ]
                            )
                            self.logger.debug(f"new computed answer: >{ac_overflow}<")

                            if ac_overflow == rep:
                                pass
                            else:
                                if self.language == "ar":
                                    final_start_positions[-1] -= 1
                                    final_end_positions[-1] -= 1

                                    ac_overflow = self.tokenizer.decode(
                                        final_input_ids[-1][
                                            final_start_positions[
                                                -1
                                            ] : final_end_positions[-1]
                                        ]
                                    )

                        else:
                            raise NotImplementedError
                    # make again the check with rep
                    if rep == ac or rep == ac_overflow:
                        self.logger.debug("\033[92mSECOND CHECK DID PASS \033[0m")
                    else:
                        self.logger.debug("\033[91mSECOND CHECK DID NOT PASS: \033[0m")
                        self.logger.debug("True text: ", rep)
                        self.logger.debug(f"answer computed ac >{ac}<")
                        self.logger.debug(
                            f"answer computed ac_overflow >{ac_overflow}<"
                        )
        self.logger.info("Done tokenizing dataset for training")
        return {
            "input_ids": final_input_ids,
            "attention_mask": final_attention_masks,
            "token_type_ids": final_token_type_ids,
            "start_positions": final_start_positions,
            "end_positions": final_end_positions,
        }
