import argparse
import os
import re
import json
import random

from tqdm import tqdm
from copy import deepcopy
import itertools

DISTRACTORS = 19

def unique_lists(mat):
    return set_lists([sorted(item) for item in mat])

def set_lists(mat):
    return [list(item) for item in set(tuple(row) for row in mat)]


def read_json(path):
    data = []
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def process_persona(args):
    random.seed(10)

    test_data = read_json(os.path.join(args.path, "dialogue2persona_test.json"))

    permutation_dialogues = []
    deletion_dialogues_1 = []
    deletion_dialogues_2 = []
    deletion_dialogues_3 = []
    addition_dialogues_1 = []
    addition_dialogues_2 = []

    for dialogue in tqdm(test_data, desc="Processing test data"):
        persona_1 = dialogue["persona_1"]
        persona_2 = dialogue["persona_2"]

        # PERMUTATION DISTRACTORS
        persona_1_permutations = list(itertools.permutations(persona_1))
        #persona_2_permutations = list(itertools.permutations(persona_2))

        for permutation_persona in persona_1_permutations:
            permutation_dialogue = deepcopy(dialogue)
            permutation_dialogue['persona_1'] = permutation_persona
            permutation_dialogues.append(permutation_dialogue)

        """for permutation_persona in persona_2_permutations:
            permutation_dialogue = deepcopy(dialogue)
            permutation_dialogue['persona_2'] = permutation_persona
            permutation_dialogues.append(permutation_dialogue)"""
    

        # DELETION DISTRACTORS (PERSONA 1)
        permute_1 = list(itertools.combinations(list(range(len(persona_1))), r=1))
        permute_2 = list(itertools.combinations(list(range(len(persona_1))), r=2))
        permute_3 = list(itertools.combinations(list(range(len(persona_1))), r=3))

        ###
        deletion_personas = []
        for indices in permute_1:
            new_persona = []

            for i in range(len(persona_1)):
                if i not in indices:
                    new_persona.append(persona_1[i])
            deletion_personas.append(new_persona)

        deletion_dialogue = deepcopy(dialogue)
        deletion_dialogue['distractors'] = deepcopy(deletion_personas)
        deletion_dialogues_1.append(deletion_dialogue)

        ###
        deletion_personas = []
        for indices in permute_2:
            new_persona = []

            for i in range(len(persona_1)):
                if i not in indices:
                    new_persona.append(persona_1[i])
            deletion_personas.append(new_persona)
        
        deletion_dialogue = deepcopy(dialogue)
        deletion_dialogue['distractors'] = deepcopy(deletion_personas)
        deletion_dialogues_2.append(deletion_dialogue)

        ###
        deletion_personas = []
        for indices in permute_3:
            new_persona = []

            for i in range(len(persona_1)):
                if i not in indices:
                    new_persona.append(persona_1[i])
            deletion_personas.append(new_persona)

        deletion_dialogue = deepcopy(dialogue)
        deletion_dialogue['distractors'] = deepcopy(deletion_personas)
        deletion_dialogues_3.append(deletion_dialogue)

        # ADDING PERSONA SENTENCES TO DISTRACTORS (PERSONA 1)
        addition_dialogue = deepcopy(dialogue)

        for i in range(len(addition_dialogue['distractors'])):
            persona_sentences = random.sample(persona_1, 1)
            addition_dialogue['distractors'][i] = persona_sentences + addition_dialogue['distractors'][i]

        addition_dialogues_1.append(addition_dialogue)

        addition_dialogue = deepcopy(dialogue)

        for i in range(len(addition_dialogue['distractors'])):
            persona_sentences = random.sample(persona_1, 2)
            addition_dialogue['distractors'][i] = persona_sentences + addition_dialogue['distractors'][i]

        addition_dialogues_2.append(addition_dialogue)

    os.makedirs(os.path.join(args.path, 'tests'), exist_ok=True)

    permutation_dialogues_path = os.path.join(args.path, 'tests', 'permutation.json')
    with open(permutation_dialogues_path, "w") as f:
        json.dump(permutation_dialogues, f, indent=4)

    deletion_dialogues_1_path = os.path.join(args.path, 'tests', 'deletion_1.json')
    with open(deletion_dialogues_1_path, "w") as f:
        json.dump(deletion_dialogues_1, f, indent=4)
    
    deletion_dialogues_2_path = os.path.join(args.path, 'tests', 'deletion_2.json')
    with open(deletion_dialogues_2_path, "w") as f:
        json.dump(deletion_dialogues_2, f, indent=4)

    deletion_dialogues_3_path = os.path.join(args.path, 'tests', 'deletion_3.json')
    with open(deletion_dialogues_3_path, "w") as f:
        json.dump(deletion_dialogues_3, f, indent=4)

    addition_dialogues_1_path = os.path.join(args.path, 'tests', 'addition_1.json')
    with open(addition_dialogues_1_path, "w") as f:
        json.dump(addition_dialogues_1, f, indent=4)

    addition_dialogues_2_path = os.path.join(args.path, 'tests', 'addition_2.json')
    with open(addition_dialogues_2_path, "w") as f:
        json.dump(addition_dialogues_2, f, indent=4)
        
    

if __name__ == "__main__":
    # ------------------------
    # ARGUMENTS
    # ------------------------
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--path", type=str, default="data/personachat/processed/both_original", help="Path to processed personachat folder.")

    args = parser.parse_args()

    process_persona(args)
