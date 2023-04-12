# Who Said That? Selecting the Correct Persona from Conversational Text

This repository provides the implementation code for the <insert_conference_name> main conference paper:

**Who Said That? Selecting the Correct Persona from Conversational Text**. [[paper]](#)


## 0. Requirements:

This project uses Python 3.6

Create a virtual env with (outside the project folder):

```bash
virtualenv -p python3.6 pers-env
source pers-env/bin/activate
```

Install the requirements (inside the project folder):
```bash
git clone git@github.com:ruinunca/selecting-the-correct-persona.git
cd selecting-the-correct-persona
pip install -r requirements.txt
```

## 1. Data Preparation

In this work, we carried out experiments using **PersonaChat**, a dataset that contains 3 to 5 profile sentences (persona) for each speaker.
We adapt the dataset in order to build the task of identifying the correct persona amongst distractors.

### Download PersonaChat

First, download PersonaChat dataset from Parlai in [this link](http://parl.ai/downloads/personachat/personachat.tgz).
Create a folder named `data/` on the project root and place it there:

```bash
mkdir data/
cd data/
wget http://parl.ai/downloads/personachat/personachat.tgz
tar -xvzf personachat.tgz
```


### Process Dataset:
```bash
python adapt_persona.py --path data/personachat --format both_original
```

Available commands:
```bash
optional arguments:
  --path                      Path to personachat folder.
  --format                    Format to be used.

formats:
  none                        Only uses dialogues.        
  self_original               Only uses persona from Speaker 1.
  self_revised                Only uses revised persona from Speaker 1.
  other_original              Only uses persona from Speaker 2.
  other_revised               Only uses revised persona from Speaker 2.
  both_original               Uses both personas.
  both_revised                Uses both revised personas.
```

Create permutations distractors for dataset:
```bash
python create_perturbations.py --path data/personachat/processed/both_original/

optional arguments:
  --path                      Path to processed dataset folder.
```

## Training

We provide scripts that can be customised.
You can either use the script:

```bash
./train.sh <gpus> <model_name_or_path> <batch_size>
```

Or run the Python file:

```bash
python run_multiple_choice.py \
--model_name_or_path google/bert_uncased_L-2_H-128_A-2 \
--task_name persona \
--output_dir experiments \
--do_eval \
--do_train \
--warmup_steps 200 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--gradient_accumulation_steps 2
```


**Note:**
After BERT several BERT-like models were released. You can test different size models like Mini-BERT and DistilBERT which are much smaller.
- Mini-BERT only contains 2 encoder layers with hidden sizes of 128 features. Use it with the flag: `--model_name_or_path google/bert_uncased_L-2_H-128_A-2`
- DistilBERT contains only 6 layers with hidden sizes of 768 features. Use it with the flag: `--model_name_or_path distilbert-base-uncased`


## Testing

### Generating the Results

First, we need to test the model by generating the results for the test set. After that, we will evaluate the results using another script:

```bash
python predict_results.py --task_name persona \
--model_name_or_path experiments/experiment_2023-03-15_14-20-49 / 
--test_file ../data/personachat/processed/both_original/tests/deletion_1.json
```

Available commands:
```bash
optional arguments:
--model_name_or_path          Path to pretrained model
--task_name                   persona (this is fixed)
--test_file                   Should contain the test data
--max_seq_length              The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
--num_distractors             The number of distractors used for training the model. (default: 9)
```

### Evaluating the Results

Now, we run a script to get the results of MRR, Accuracy, and other metrics that could be added:

```bash
python evaluate_results.py --path experiments/experiment_2023-03-15_14-20-49/results/
```
