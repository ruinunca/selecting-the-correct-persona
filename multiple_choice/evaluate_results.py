import argparse
import os
import re
import json
import random
import numpy as np

from pathlib import Path

from tqdm import tqdm
from copy import deepcopy
import itertools


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
    

def read_json(path):
    data = []
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def process_results(args):
    test_data = read_json(args.path)

    hits_1, hits_2, hits_3, hits_4, hits_5, hits_10 = [], [], [], [], [], []
    mrr = []

    examples = {'metrics': {}, 'incorrect': [], 'correct': []}

    for example in test_data:
        prediction = example['prediction']['id']
        label = example['label']['id']
        rankings = example["ranking"]

        hits_1.append(label in rankings[:1])
        hits_2.append(label in rankings[:2])
        hits_3.append(label in rankings[:3])
        hits_4.append(label in rankings[:4])
        hits_5.append(label in rankings[:5])
        hits_10.append(label in rankings[:10])

        rankings = np.array(rankings)

        rank = label == rankings
        rank = rank.astype(int)  
        mrr.append(rank)

        if prediction == label:
            examples['correct'].append(example)
        else:
            examples['incorrect'].append(example)

    examples['metrics']['hits_1'] = sum(hits_1) / len(hits_1)
    examples['metrics']['hits_2'] = sum(hits_2) / len(hits_2)
    examples['metrics']['hits_3'] = sum(hits_3) / len(hits_3)
    examples['metrics']['hits_4'] = sum(hits_4) / len(hits_4)
    examples['metrics']['hits_5'] = sum(hits_5) / len(hits_5)
    examples['metrics']['hits_10'] = sum(hits_10) / len(hits_10)
    examples['metrics']['mrr'] = mean_reciprocal_rank(mrr)

    print("Hits@1: {}".format(examples['metrics']['hits_1']))
    print("Hits@2: {}".format(examples['metrics']['hits_2']))
    print("Hits@3: {}".format(examples['metrics']['hits_3']))
    print("Hits@4: {}".format(examples['metrics']['hits_4']))
    print("Hits@5: {}".format(examples['metrics']['hits_5']))
    print("Hits@10: {}".format(examples['metrics']['hits_10']))
    print("MRR: {}".format(examples['metrics']['mrr']))

    path = Path(args.path)
    name = path.stem   
    parent = path.parent

    results_path = os.path.join(parent, "scores_" + name + ".json")

    with open(results_path, "w") as f:
        json.dump(examples, f, indent=4)
    
    print("Saved to:", results_path)
    

if __name__ == "__main__":
    # ------------------------
    # ARGUMENTS
    # ------------------------
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--path", type=str, help="Path to processed personachat folder.", required=True)

    args = parser.parse_args()

    process_results(args)
