import json
import logging
import os
import torch
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import tqdm
from filelock import FileLock

from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available

from torch.utils.data.dataset import Dataset

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class InputExample:
    """
    A single training/test example for multiple choice
    Args:
        example_id: Unique id for the example.
        question: string. The untokenized text of the second sequence (question).
        contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
        endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    example_id: str
    question: str
    contexts: List[str]
    endings: List[str]
    label: Optional[str]


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    label: Optional[int]


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class MultipleChoiceDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeatures]

    def __init__(
        self,
        data_dir: str,
        tokenizer: PreTrainedTokenizer,
        task: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
        mode: Split = Split.train,
    ):
        processor = processors[task]()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length), task,),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                logger.info(f"Loading features from cached file {cached_features_file}")
                self.features = torch.load(cached_features_file)
            else:
                logger.info(f"Creating features from dataset file at {data_dir}")
                label_list = processor.get_labels()
                if mode == Split.dev:
                    examples = processor.get_dev_examples(data_dir)
                elif mode == Split.test:
                    examples = processor.get_test_examples(data_dir)
                else:
                    examples = processor.get_train_examples(data_dir)
                logger.info("Training examples: %s", len(examples))
                # TODO clean up all this to leverage built-in features of tokenizers
                self.features = convert_examples_to_features(
                    examples,
                    label_list,
                    max_seq_length,
                    tokenizer,
                    pad_on_left=bool(tokenizer.padding_side == "left"),
                    pad_token=tokenizer.pad_token_id,
                    pad_token_segment_id=tokenizer.pad_token_type_id,
                )
                logger.info("Saving features into cached file %s", cached_features_file)
                torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class PersonaProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        path = os.path.join(data_dir, "dialogue2persona_train.json")
        data = self._read_json(path)
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        path = os.path.join(data_dir, "dialogue2persona_valid.json")
        data = self._read_json(path)
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        path = os.path.join(data_dir, "dialogue2persona_test.json")
        data = self._read_json(path)
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return list(range(MULTIPLE_CHOICE_TASKS_NUM_LABELS["persona"]))

    def _read_json(self, path):
        data = []
        with open(path, 'r') as f:
            data = json.load(f)
        return data

    def _create_examples(self, dialogues_raw, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        candidates =  MULTIPLE_CHOICE_TASKS_NUM_LABELS['persona']
        distractors = candidates - 1
        
        for i, dial_raw in enumerate(dialogues_raw):
            turns_1 = [turn['speaker_1'] for turn in dial_raw['turns']]
            turns_2 = [turn['speaker_2'] for turn in dial_raw['turns']]
            
            personas = [" ".join(dial_raw['persona_1'])] + [" ".join(persona) for persona in dial_raw['distractors'][:distractors]]
            dialogues = [" ".join(turns_1)] * candidates


            examples.append(
                InputExample(
                    example_id=i, # FIX: is it necessary?
                    question="",
                    contexts=dialogues,  # this is not efficient but convenient
                    endings=personas,
                    label=0, # It is always the first one
                )
            )
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm.tqdm(examples, desc="Convert examples to features")):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_inputs = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                pad_to_max_length=True,
                return_overflowing_tokens=True,
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            choices_inputs.append(inputs)

        label = label_map[example.label]

        input_ids = [x["input_ids"] for x in choices_inputs]
        attention_mask = (
            [x["attention_mask"] for x in choices_inputs] if "attention_mask" in choices_inputs[0] else None
        )
        token_type_ids = (
            [x["token_type_ids"] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
        )

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
            )
        )

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


processors = {"persona": PersonaProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"persona": 10}
