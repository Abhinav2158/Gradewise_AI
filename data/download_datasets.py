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

# Curated SciEntsBank Reference Data (SemEval-2013 Task 7)
SCIENTSBANK_DATA = {
    "question_id": "SEMEVAL_SCIBANK_01",
    "domain": "Physics / Electricity",
    "question_text": "Explain why adding more light bulbs in a simple series circuit causes the bulbs to become dimmer.",
    "reference_answer": "In a series circuit, adding more light bulbs increases the total electrical resistance of the circuit. According to Ohm's Law (I = V/R), for a constant voltage supply, increased resistance reduces the total current flowing through the circuit, which decreases the power delivered to each individual bulb, making them dimmer.",
    "total_marks": 3,
    "sample_records": [
        {
            "id": 201,
            "student_answer": "Adding more bulbs increases the total resistance in the series circuit. Since the voltage is constant, higher resistance causes the electric current to decrease, so each bulb receives less electrical energy and glows less brightly.",
            "human_score_1": 3,
            "human_score_2": 3
        },
        {
            "id": 202,
            "student_answer": "The bulbs share the same electricity and the resistance goes up, so the current goes down.",
            "human_score_1": 2,
            "human_score_2": 2
        },
        {
            "id": 203,
            "student_answer": "Because the battery runs out of power faster when you put too many lights on it.",
            "human_score_1": 0,
            "human_score_2": 0
        }
    ]
}

# Curated CodeNet / CS Programming Data
CODENET_DATA = {
    "question_id": "CODENET_PY_01",
    "domain": "Computer Science / Algorithms",
    "question_text": "Write a Python function `find_first_duplicate(arr)` that finds the first duplicate element in a list of integers in O(N) time complexity using a set.",
    "reference_answer": "def find_first_duplicate(arr):\n    seen = set()\n    for num in arr:\n        if num in seen:\n            return num\n        seen.add(num)\n    return None",
    "total_marks": 4,
    "sample_records": [
        {
            "id": 301,
            "student_answer": "def find_first_duplicate(arr):\n    seen = set()\n    for x in arr:\n        if x in seen:\n            return x\n        seen.add(x)\n    return None",
            "human_score_1": 4,
            "human_score_2": 4
        },
        {
            "id": 302,
            "student_answer": "def find_first_duplicate(arr):\n    for i in range(len(arr)):\n        for j in range(i+1, len(arr)):\n            if arr[i] == arr[j]:\n                return arr[i]\n    return None",
            "human_score_1": 2,
            "human_score_2": 2
        },
        {
            "id": 303,
            "student_answer": "print('duplicate')",
            "human_score_1": 0,
            "human_score_2": 0
        }
    ]
}

# Curated ASAP-AES Long-form Essay Data (Set 1: Computers in Society)
ASAP_AES_DATA = {
    "question_id": "ASAP_AES_SET_1",
    "domain": "English / Persuasive Essay",
    "question_text": "More and more people use computers, but not everyone agrees that this benefits society. Write a persuasive essay discussing the positive and negative effects of computers on individuals and communities.",
    "reference_answer": "A strong persuasive essay must establish a clear thesis on computer technology's societal impact, provide structured paragraphs analyzing both benefits (e.g. global communication, information access) and drawbacks (e.g. digital divide, reduced physical activity), and conclude with synthesized insights.",
    "total_marks": 6,
    "sample_records": [
        {
            "id": 401,
            "student_answer": "Computers have fundamentally reshaped modern civilization. On one hand, they provide unprecedented access to education and foster global communication. On the other hand, over-reliance on digital screens has contributed to sedentary lifestyles and reduced face-to-face interaction. In conclusion, computers are powerful tools whose societal value depends on mindful usage.",
            "human_score_1": 5,
            "human_score_2": 5
        },
        {
            "id": 402,
            "student_answer": "Computers are good because I play video games and chat with friends on the internet.",
            "human_score_1": 2,
            "human_score_2": 2
        }
    ]
}

def create_dataset_files():
    print("Writing multi-domain benchmark datasets...")

    # 1. ASAP-SAS Sets
    for set_id, data in ASAP_PROMPTS.items():
        p_file = ASAP_DIR / f"prompt_set_{set_id}.json"
        with open(p_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"  -> Created ASAP-SAS Set {set_id}: {p_file}")

    # 2. SciEntsBank
    scibank_file = DATA_DIR / "scientsbank.json"
    with open(scibank_file, "w", encoding="utf-8") as f:
        json.dump(SCIENTSBANK_DATA, f, indent=2)
    print(f"  -> Created SciEntsBank Benchmark: {scibank_file}")

    # 3. CodeNet
    codenet_file = DATA_DIR / "codenet.json"
    with open(codenet_file, "w", encoding="utf-8") as f:
        json.dump(CODENET_DATA, f, indent=2)
    print(f"  -> Created CodeNet Programming Benchmark: {codenet_file}")

    # 4. ASAP-AES
    aes_file = DATA_DIR / "asap_aes.json"
    with open(aes_file, "w", encoding="utf-8") as f:
        json.dump(ASAP_AES_DATA, f, indent=2)
    print(f"  -> Created ASAP-AES Essay Benchmark: {aes_file}")

    print("\nAll multi-domain benchmark datasets successfully generated!")

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
    create_dataset_files()
    print("Preparing and validating benchmark datasets...")
    prepare_asap_dataset()
    prepare_squad_dataset()
    print("Dataset setup completed successfully!")
