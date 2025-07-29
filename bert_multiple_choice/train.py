import argparse
import json, os

from data_module import DataCollatorForMultipleChoice
from tqdm import tqdm
from transformers import (
    AutoModelForMultipleChoice, 
    TrainingArguments, 
    Trainer, 
    AutoTokenizer
)


def preprocess_function(dialogues, distractors, tokenizer):
    candidates = distractors + 1 # true sentence
    turns_1 = [[turn['speaker_1'] for turn in dialogue['turns']] for dialogue in dialogues]
    turns_2 = [[turn['speaker_2'] for turn in dialogue['turns']] for dialogue in dialogues]

    turns = [[" ".join(history)] * candidates for history in turns_1]
    personas = [[" ".join(dialogue['persona_1'])] + [" ".join(persona) for persona in dialogue['distractors'][:distractors]] for dialogue in dialogues] 

    # Flatten everything
    turns = sum(turns, [])
    personas = sum(personas, [])

    tokenized_examples = tokenizer(turns, personas, truncation=True)

    return_dict = {k: [v[i : i + candidates] for i in range(0, len(v), candidates)] for k, v in tokenized_examples.items()}
    return return_dict


def main(hparams):
    # Load dataset

    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2")

    tokenized_dataset = {}

    for split in ['train', 'valid', 'test']:
        path = os.path.join(hparams.data_path, "dialogue2persona_" + split + ".json")

        with open(path, 'r') as f:
            data = json.load(f)

        tokenized_dataset[split] = []

        for dialogue in tqdm(data[:10], desc=f"Processing {split} data"):
            tokenized_dataset[split].append(preprocess_function(dialogue, 10, tokenizer))
    
    #print(tokenized_dataset[split][0])
    #print([tokenizer.decode(tokenized_dataset[split][0]["input_ids"][3][i]) for i in range(10)])
    

    #model = AutoModelForMultipleChoice.from_pretrained("bert-base-uncased") 
    model = AutoModelForMultipleChoice.from_pretrained("google/bert_uncased_L-2_H-128_A-2")
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=hparams.learning_rate,
        per_device_train_batch_size=hparams.train_batch_size,
        per_device_eval_batch_size=hparams.valid_batch_size,
        num_train_epochs=hparams.epochs,
        weight_decay=hparams.weight_decay,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["valid"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    )

    trainer.train()



if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument(
        "--data_path",
        default="../data/personachat/processed/both_original",
        type=str,
    )

    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The best k models according to the quantity monitored will be saved.",
    )
    parser.add_argument(
        "--epochs",
        default=3,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--train_batch_size", default=16, type=int, help="Train batch size to be used."
    )
    parser.add_argument(
        "--valid_batch_size", default=16, type=int, help="Train batch size to be used."
    )
    parser.add_argument(
        "--weight_decay",
        default=0.01,
        type=float,
    )

    # gpu args
    parser.add_argument("--gpus", type=int, default=1, help="How many gpus")
    parser.add_argument(
        "--val_check_interval",
        default=1.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_acc", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=2,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)
        