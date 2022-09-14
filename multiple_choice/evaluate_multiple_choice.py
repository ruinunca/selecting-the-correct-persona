# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""


import enum
import logging
import os, json
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from utils import MultipleChoiceDataset, Split, processors


logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(
        metadata={"help": "Should contain the data files for the task."},
        default="../data/personachat/processed/both_original"
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Set seed
    set_seed(42)

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForMultipleChoice.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    # Get datasets
    """eval_dataset = MultipleChoiceDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        task=data_args.task_name,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.dev,
    )"""
    test_dataset = MultipleChoiceDataset(
        data_dir=data_args.data_dir,
        tokenizer=tokenizer,
        task=data_args.task_name,
        max_seq_length=data_args.max_seq_length,
        overwrite_cache=data_args.overwrite_cache,
        mode=Split.test,
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        preds = np.argmax(p.predictions, axis=1)
        return {"acc": simple_accuracy(preds, p.label_ids)}

    # Initialize our Trainer
    """trainer = Trainer(
        model=model,
        eval_dataset=eval_dataset,

        compute_metrics=compute_metrics,
    )

    # Evaluation
    results = {}
    logger.info("*** Evaluate ***")

    result = trainer.evaluate()

    output_eval_file = os.path.join(model_args.model_name_or_path, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key, value in result.items():
            logger.info("  %s = %s", key, value)
            writer.write("%s = %s\n" % (key, value))

        results.update(result)"""

    
    # Test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    correct, total_examples = 0, 0

    examples = {'accuracy': 0, 'incorrect': [], 'correct': []}

    for dialogue in tqdm(test_dataset, desc="Evaluating test data"):
        input_ids = torch.tensor(dialogue.input_ids).to(device)
        token_type_ids = torch.tensor(dialogue.token_type_ids).to(device)
        attention_mask = torch.tensor(dialogue.attention_mask).to(device) 
        label = dialogue.label

        with torch.no_grad():
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        logits = outputs.logits.detach().cpu().numpy()
        
        prediction = np.argmax(logits).item()
        probabilities = F.softmax(outputs.logits, dim=0).detach().cpu().tolist()
        probabilities = [item[0] for item in probabilities] # list of 1 element
        probabilities = {k: v for k,v in enumerate(probabilities)}

        """choices = {}
        for i, choice in enumerate(dialogue.input_ids):
            choices[i] = tokenizer.decode(choice[0])

        example = {
            'choices': choices,
            'label': label, 
            'prediction': prediction,
        }"""

        example = {
            'label': {
                'id': label,
                'text': tokenizer.decode(dialogue.input_ids[label][0])
            },
            'prediction': {
                'id': prediction,
                'text': tokenizer.decode(dialogue.input_ids[prediction][0]),
            },
            'distribution': probabilities,
        }

        if prediction == label:
            correct += 1
            examples['correct'].append(example)
        else:
            examples['incorrect'].append(example)

        total_examples += 1

    accuracy = correct / total_examples * 100
    
    print(f"Accuracy: {accuracy}%")
    examples['accuracy'] = accuracy

    with open(os.path.join(model_args.model_name_or_path, "test_examples.json"), "w") as f:
        json.dump(examples, f, indent=4)
    
    """predictions = trainer.predict(test_dataset=test_dataset)
    print(predictions.predictions.shape, predictions.label_ids.shape)
    preds = np.argmax(predictions.predictions, axis=-1)

    acc = np.sum(preds == predictions.label_ids) / len(predictions.label_ids)
    print(acc)"""

    return 0


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()