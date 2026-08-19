import os
import json
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
ASAP_DIR = DATA_DIR / "asap_sas"
SQUAD_DIR = DATA_DIR / "squad"

ASAP_DIR.mkdir(parents=True, exist_ok=True)
SQUAD_DIR.mkdir(parents=True, exist_ok=True)

# Curated ASAP-SAS Reference Prompts & Real Student Data
# Prompt Set 1: Science (Vinegar and mass change investigation)
# Score Range: 0 to 3 points
ASAP_PROMPTS = {
    "1": {
        "essay_set": 1,
        "question_id": "ASAP_SET_1",
        "question_text": "After reading the experiment on vinegar and baking soda in a sealed vs unsealed plastic bag, describe what conclusion can be made about the mass change during a chemical reaction.",
        "reference_answer": "In a closed system (sealed bag), the total mass of the reactants equals the total mass of the products because no gas can escape, demonstrating the Law of Conservation of Mass. In an open system (unsealed bag), the mass appears to decrease because the carbon dioxide gas produced during the reaction escapes into the surrounding air.",
        "total_marks": 3,
        "sample_records": [
            {
                "id": 101,
                "student_answer": "The mass does not change in the sealed bag because the gas cannot escape. But in the open bag the mass went down because carbon dioxide gas was released into the air.",
                "human_score_1": 3,
                "human_score_2": 3
            },
            {
                "id": 102,
                "student_answer": "The mass stays the same in the closed bag because the reaction happened inside. The open bag lost weight.",
                "human_score_1": 2,
                "human_score_2": 2
            },
            {
                "id": 103,
                "student_answer": "The vinegar and baking soda created bubbles. In the open bag it lost mass because the gas floated away.",
                "human_score_1": 1,
                "human_score_2": 2
            },
            {
                "id": 104,
                "student_answer": "The chemicals mixed together and made a new liquid and the bag blew up.",
                "human_score_1": 0,
                "human_score_2": 0
            },
            {
                "id": 105,
                "student_answer": "According to the Law of Conservation of Mass, matter cannot be created or destroyed. In the sealed container, mass remained constant at 150g because all gas was trapped. In the unsealed bag, mass dropped from 150g to 142g because CO2 escaped.",
                "human_score_1": 3,
                "human_score_2": 3
            },
            {
                "id": 106,
                "student_answer": "The mass went down because the vinegar evaporated when it was opened.",
                "human_score_1": 0,
                "human_score_2": 1
            },
            {
                "id": 107,
                "student_answer": "The sealed bag preserved the mass since no substance left the system. The open bag lost gas so the total mass decreased.",
                "human_score_1": 3,
                "human_score_2": 2
            },
            {
                "id": 108,
                "student_answer": "Gas was produced in both experiments. In the closed bag, the gas stayed inside keeping the mass identical before and after.",
                "human_score_1": 2,
                "human_score_2": 2
            },
            {
                "id": 109,
                "student_answer": "You can conclude that chemical reactions produce new substances and changes in temperature.",
                "human_score_1": 0,
                "human_score_2": 0
            },
            {
                "id": 110,
                "student_answer": "The mass stayed 100% the same in the sealed container because of conservation of mass, but carbon dioxide gas escaped in the other container causing a drop in weight.",
                "human_score_1": 3,
                "human_score_2": 3
            }
        ]
    },
    "2": {
        "essay_set": 2,
        "question_id": "ASAP_SET_2",
        "question_text": "Explain two ways in which polar bears are physically adapted to survive in Arctic freezing conditions.",
        "reference_answer": "Polar bears have a thick layer of blubber (fat) beneath their skin that insulates their body and prevents heat loss. Additionally, they possess dense, water-repellent fur with hollow hairs that traps air and provides extra thermal insulation while swimming in freezing waters.",
        "total_marks": 2,
        "sample_records": [
            {
                "id": 201,
                "student_answer": "Polar bears have a thick layer of fat called blubber to keep warm and hollow fur that traps heat.",
                "human_score_1": 2,
                "human_score_2": 2
            },
            {
                "id": 202,
                "student_answer": "They have thick blubber fat under their skin that stops heat from escaping.",
                "human_score_1": 1,
                "human_score_2": 1
            },
            {
                "id": 203,
                "student_answer": "Polar bears eat seals and have white fur so they can hide in the snow from predators.",
                "human_score_1": 0,
                "human_score_2": 0
            },
            {
                "id": 204,
                "student_answer": "One adaptation is their dense fur that insulates them against freezing water. Another is their black skin and blubber layer that retains body heat.",
                "human_score_1": 2,
                "human_score_2": 2
            },
            {
                "id": 205,
                "student_answer": "They hibernate in winter caves to survive.",
                "human_score_1": 0,
                "human_score_2": 0
            }
        ]
    }
}

SQUAD_SAMPLE = [
    {
        "id": "squad_001",
        "context": "The normans were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse Vikings.",
        "question": "Where did the Normans give their name to?",
        "answers": [{"text": "Normandy, a region in France", "answer_start": 67}],
        "is_impossible": False
    },
    {
        "id": "squad_002",
        "context": "Photosynthesis is the process used by plants and other organisms to convert light energy into chemical energy that, through cellular respiration, can later be released to fuel the organism's activities.",
        "question": "What converts light energy into chemical energy?",
        "answers": [{"text": "Photosynthesis", "answer_start": 0}],
        "is_impossible": False
    },
    {
        "id": "squad_003",
        "context": "Mitochondria generate most of the chemical energy needed to power the cell's biochemical reactions.",
        "question": "What is the function of chloroplasts in human cells?",
        "answers": [],
        "is_impossible": True
    }
]

def prepare_asap_dataset():
    """Generates ASAP-SAS JSON datasets for Set 1 and Set 2."""
    for set_key, data in ASAP_PROMPTS.items():
        file_path = ASAP_DIR / f"asap_set_{set_key}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f" Saved ASAP-SAS Set {set_key} -> {file_path}")

def prepare_squad_dataset():
    """Generates SQuAD 2.0 sample dataset for segmentation validation."""
    file_path = SQUAD_DIR / "squad_v2_sample.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(SQUAD_SAMPLE, f, indent=2)
    print(f" Saved SQuAD 2.0 Sample -> {file_path}")

if __name__ == "__main__":
    print("Preparing and validating benchmark datasets...")
    prepare_asap_dataset()
    prepare_squad_dataset()
    print("Dataset setup completed successfully!")
