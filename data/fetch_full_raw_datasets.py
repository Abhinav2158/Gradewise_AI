"""
Unified Raw Dataset Downloader & Loader for Full Corpus Fine-Tuning & Benchmarks
Datasets supported:
1. SQuAD 2.0 (150,000+ extractive QA pairs)
2. SciEntsBank / SemEval-2013 Task 7 (10,000+ Science QA pairs)
3. ASAP-SAS (Short Answer Scoring - 17,200+ student responses across 10 prompts)
4. ASAP-AES (Automated Essay Scoring - 13,000+ long persuasive essays across 8 prompts)
5. CodeNet / HumanEval (Python code logic and algorithmic reasoning)
"""

import os
import sys
import json
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_squad_full():
    print("\n[1/5] Downloading Full SQuAD 2.0 Corpus...")
    dest = RAW_DIR / "squad_v2_full.json"
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
    try:
        if not dest.exists():
            print(f"  -> Fetching from {url}...")
            urllib.request.urlretrieve(url, dest)
            print(f"  -> Saved {dest.name} ({os.path.getsize(dest) / (1024*1024):.2f} MB)")
        else:
            print(f"  -> File already exists: {dest}")
    except Exception as e:
        print(f"  [!] Notice: SQuAD download error ({e}). Trying datasets library fallback...")
        try:
            from datasets import load_dataset
            ds = load_dataset("rajpurkar/squad_v2")
            ds.save_to_disk(str(RAW_DIR / "squad_v2_hf"))
            print(f"  -> Successfully saved via HuggingFace datasets to {RAW_DIR / 'squad_v2_hf'}")
        except Exception as err:
            print(f"  [!] Fallback notice: {err}")

def download_scientsbank_full():
    print("\n[2/5] Downloading SciEntsBank (SemEval-2013 Task 7 Full Corpus)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("sem_eval_2013_task_7", "2-way", trust_remote_code=True)
        out_dir = RAW_DIR / "scientsbank_full"
        ds.save_to_disk(str(out_dir))
        print(f"  -> Successfully saved SciEntsBank to {out_dir}")
    except Exception as e:
        print(f"  [!] SciEntsBank Hugging Face notice ({e}).")

def download_humaneval_codenet():
    print("\n[3/5] Downloading CodeNet / HumanEval Python Benchmark...")
    dest = RAW_DIR / "humaneval_full.jsonl"
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    try:
        import gzip
        gz_dest = RAW_DIR / "HumanEval.jsonl.gz"
        if not gz_dest.exists():
            print(f"  -> Fetching from {url}...")
            urllib.request.urlretrieve(url, gz_dest)
            with gzip.open(gz_dest, 'rb') as f_in:
                with open(dest, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"  -> Extracted HumanEval Python benchmark to {dest}")
        else:
            print(f"  -> File already exists: {dest}")
    except Exception as e:
        print(f"  [!] HumanEval download notice: {e}")

def download_asap_sas_full():
    print("\n[4/5] Downloading ASAP-SAS Full Short Answer Corpus (10 Prompt Sets)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("asap_sas", trust_remote_code=True)
        out_dir = RAW_DIR / "asap_sas_full"
        ds.save_to_disk(str(out_dir))
        print(f"  -> Successfully saved full ASAP-SAS to {out_dir}")
    except Exception as e:
        print(f"  [!] Note on ASAP-SAS: Kaggle terms may require Kaggle API key (kaggle competitions download -c asap-sas).")

def download_asap_aes_full():
    print("\n[5/5] Downloading ASAP-AES Full Automated Essay Scoring Corpus (8 Prompt Sets)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("aes_dataset", trust_remote_code=True)
        out_dir = RAW_DIR / "asap_aes_full"
        ds.save_to_disk(str(out_dir))
        print(f"  -> Successfully saved full ASAP-AES to {out_dir}")
    except Exception as e:
        print(f"  [!] Note on ASAP-AES: Kaggle terms may require Kaggle API key (kaggle competitions download -c asap-aes).")

def main():
    print("=================================================================")
    print("       FULL-SCALE RAW DATASET DOWNLOADER & LOADER PIPELINE       ")
    print("=================================================================")
    print(f"Target Directory: {RAW_DIR}")

    download_squad_full()
    download_scientsbank_full()
    download_humaneval_codenet()
    download_asap_sas_full()
    download_asap_aes_full()

    print("\n=================================================================")
    print("Dataset download pipeline finished. Check data/raw/ directory.")
    print("=================================================================")

if __name__ == "__main__":
    main()
