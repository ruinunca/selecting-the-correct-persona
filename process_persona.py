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

from tqdm import tqdm


def process_persona(args):
    SPEAKER_2 = "your persona: "
    SPEAKER_1 = "partner's persona: "

    for split in ['train', 'valid', 'test']:
        path = os.path.join(args.path, split + "_" + args.format + ".txt")
        with open(path) as f:
            lines = f.readlines()
        last_number = 0

        dialogues = []
        dialogue = {}
        persona_1, persona_2 = [], []
        turns = []

        for line in tqdm(lines, desc=f"Processing {split} split"):
            line = line.strip().replace('\\n', '\n')
            line_number = int(re.findall(r"\d+", line)[0])

            # New dialogue: save dialogue and reset everything
            if line_number < last_number:
                dialogue['persona_1'] = persona_1
                dialogue['persona_2'] = persona_2
                dialogue['turns'] = turns
                
                dialogues.append(dialogue)
                # Reset
                persona_1, persona_2 = [], []
                dialogue = {}
                turns = []

            line = line[2:] # Remove number
            
            if SPEAKER_1 in line:
                persona_text = line.replace(SPEAKER_1, "")
                persona_1.append(persona_text)
            elif SPEAKER_2 in line:
                persona_text = line.replace(SPEAKER_2, "")
                persona_2.append(persona_text)
            else:
                turn = {}
                sentences = line.split('\t')
                candidates = "".join(sentences[-1]).split('|')

                turn['speaker_1'] = sentences[0]
                # Last candidate is the correct answer
                turn['speaker_2'] = candidates[-1]
                turn['distractors'] = candidates[:-1]

                turns.append(turn)

            last_number = line_number
        
        processed_dir = os.path.join(args.path, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        process_path = os.path.join(processed_dir, split + '_' + args.format + '.json')

        with open(process_path, "w") as f:
            json.dump(dialogues, f, indent=4)



if __name__ == "__main__":
    # ------------------------
    # ARGUMENTS
    # ------------------------
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--format", required=True, type=str, help="Format of output.", choices=['none', 'self_original', 'self_revised', 'other_original', 'other_revised', 'both_original', 'both_revised'])
    parser.add_argument("--path", type=str, default="data/personachat", help="Path to personachat folder.")

    args = parser.parse_args()

    process_persona(args)
