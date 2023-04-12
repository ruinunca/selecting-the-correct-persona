"""
Process personachat dataset.

Persona can be 'none', 'self', 'other', or 'both'.
Format of persona can be 'original' or 'revised'.

--format personachat:{format}
...where {format} is one of...
- none
- self_original
- self_revised
- other_original
- other_revised
- both_original
- both_revised

(Inspiration from ParlAI)
"""
import argparse
import os
import re
import json
import random

from tqdm import tqdm
from copy import deepcopy

DISTRACTORS = 19

def unique_lists(mat):
    return set_lists([sorted(item) for item in mat])

def set_lists(mat):
    return [list(item) for item in set(tuple(row) for row in mat)]


def process_persona(args):
    random.seed(10)

    SPEAKER_2 = "your persona: "
    SPEAKER_1 = "partner's persona: "

    processed_dir = os.path.join(args.path, "processed", args.format)
    os.makedirs(processed_dir, exist_ok=True)

    all_personas_dict = {}
    all_persona_sentences_dict = {}
    dataset2persona = {}

    for split in ['train', 'valid', 'test']:
        # Statistics
        all_persona_sentences = []
        all_personas_1 = []
        all_personas_2 = []

        # Read file
        path = os.path.join(args.path, split + "_" + args.format + ".txt")
        with open(path) as f:
            lines = f.readlines()
        last_number = 0

        dialogues = []
        dialogue = {}
        persona_1, persona_2 = [], []
        turns = []

        # dialogue-to-persona
        dialogues2persona = []
        dialogue2persona = {}
        turns2persona = []

        for line in tqdm(lines, desc=f"Processing {split} split"):
            line = line.strip().replace('\\n', '\n')
            line_number = int(re.findall(r"\d+", line)[0])

            # New dialogue: save dialogue and reset everything
            if line_number < last_number:
                dialogue['persona_1'] = persona_1
                dialogue['persona_2'] = persona_2
                dialogue['turns'] = turns
                dialogues.append(dialogue)

                # dialogue-to-persona
                dialogue2persona['persona_1'] = persona_1
                dialogue2persona['persona_2'] = persona_2
                dialogue2persona['turns'] = turns2persona
                dialogues2persona.append(dialogue2persona)

                # Statistics
                all_personas_1.append(persona_1)
                all_personas_2.append(persona_2)
                # Reset
                persona_1, persona_2 = [], []
                dialogue, dialogue2persona = {}, {}
                turns, turns2persona = [], []

            line = line[2:] # Remove number
            
            if SPEAKER_1 in line:
                persona_text = line.replace(SPEAKER_1, "").lstrip().rstrip()
                persona_1.append(persona_text)

                all_persona_sentences.append(persona_text)
            elif SPEAKER_2 in line:
                persona_text = line.replace(SPEAKER_2, "").lstrip().rstrip()
                persona_2.append(persona_text)

                all_persona_sentences.append(persona_text)
            else:
                turn = {}
                sentences = line.split('\t')
                candidates = "".join(sentences[-1]).split('|')

                turn['speaker_1'] = sentences[0].lstrip().rstrip()

                # Last candidate is the correct answer
                turn['speaker_2'] = candidates[-1].lstrip().rstrip()

                turn2persona = deepcopy(turn)
                turn['distractors'] = candidates[:-1]

                turns.append(turn)
                turns2persona.append(turn2persona)

            last_number = line_number

        process_path = os.path.join(processed_dir, split + '.json')

        with open(process_path, "w") as f:
            json.dump(dialogues, f, indent=4)

        
        dataset2persona[split] = dialogues2persona


        # Statistics
        all_persona_sentences = list(set(all_persona_sentences))
        print("* Statistics:")
        print(f"Count of different persona sentences: {len(all_persona_sentences)}")

        all_personas = all_personas_1 + all_personas_2
        all_personas_1 = unique_lists(all_personas_1)
        all_personas_2 = unique_lists(all_personas_2)
        all_personas = unique_lists(all_personas)
        print(f"Count of different complete personas (speaker 1): {len(all_personas_1)}")
        print(f"Count of different complete personas (speaker 2): {len(all_personas_2)}")
        print(f"Count of different complete personas (all): {len(all_personas)}")
        print()

        all_personas_dict[split] = all_personas
        all_persona_sentences_dict[split] = all_persona_sentences

    
    personas_path = os.path.join(processed_dir, 'personas.json')
    persona_sentences_path = os.path.join(processed_dir, 'persona_sentences.json')

    with open(personas_path, "w") as f:
        json.dump(all_personas_dict, f, indent=4)

    with open(persona_sentences_path, "w") as f:
        json.dump(all_persona_sentences_dict, f, indent=4)

    for split, dialogues2persona in dataset2persona.items():
        
        for dialogue in dialogues2persona:
            found_duplicate = 1
            while found_duplicate:
                distractors = random.sample(all_personas, DISTRACTORS)
                if not dialogue['persona_1'] in distractors and not dialogue['persona_2'] in distractors:
                    found_duplicate = 0
                
            dialogue['distractors'] = distractors

        dialogue2persona_path = os.path.join(processed_dir, 'dialogue2persona_' + split + '.json')

        with open(dialogue2persona_path, "w") as f:
            json.dump(dialogues2persona, f, indent=4)
        
    

if __name__ == "__main__":
    # ------------------------
    # ARGUMENTS
    # ------------------------
    parser = argparse.ArgumentParser(
        description="Adapt persona for the selection task.",
        add_help=True,
    )
    parser.add_argument("--format", required=True, type=str, help="Format of output.", choices=['none', 'self_original', 'self_revised', 'other_original', 'other_revised', 'both_original', 'both_revised'])
    parser.add_argument("--path", type=str, default="data/personachat", help="Path to personachat folder.")

    args = parser.parse_args()

    process_persona(args)
