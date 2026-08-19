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
import gzip
import zipfile
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

def setup_kaggle_auth():
    """Ensures Kaggle API credentials are authenticated."""
    token = os.environ.get("KAGGLE_API_TOKEN", "KGAT_99eca7fce7b6382e4fab8b1baa8afdae")
    os.environ["KAGGLE_API_TOKEN"] = token
    
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_file = kaggle_dir / "access_token"
    token_file.write_text(token)
    
    try:
        import kaggle
        kaggle.api.authenticate()
        print("  -> Kaggle API authenticated successfully.")
        return True
    except Exception as e:
        print(f"  [!] Kaggle authentication warning: {e}")
        return False

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
            print(f"  -> File already exists: {dest.name} ({os.path.getsize(dest) / (1024*1024):.2f} MB)")
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
    out_dir = RAW_DIR / "scientsbank_full"
    try:
        if out_dir.exists() and any(out_dir.iterdir()):
            print(f"  -> SciEntsBank full dataset already downloaded in {out_dir}")
            return
        from datasets import load_dataset
        ds = load_dataset("nkazi/SciEntsBank")
        ds.save_to_disk(str(out_dir))
        print(f"  -> Successfully saved SciEntsBank dataset to {out_dir}")
    except Exception as e:
        print(f"  [!] SciEntsBank notice: {e}")

def download_humaneval_codenet():
    print("\n[3/5] Downloading CodeNet / HumanEval Python Benchmark...")
    dest = RAW_DIR / "humaneval_full.jsonl"
    gz_dest = RAW_DIR / "HumanEval.jsonl.gz"
    url = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
    try:
        if not dest.exists():
            print(f"  -> Fetching from {url}...")
            urllib.request.urlretrieve(url, gz_dest)
            with gzip.open(gz_dest, 'rb') as f_in:
                with open(dest, 'wb') as f_out:
                    f_out.write(f_in.read())
            print(f"  -> Extracted HumanEval Python benchmark to {dest.name}")
        else:
            print(f"  -> File already exists: {dest.name}")
    except Exception as e:
        print(f"  [!] HumanEval download notice: {e}")

def download_asap_sas_full():
    print("\n[4/5] Downloading ASAP-SAS Full Short Answer Corpus (10 Prompt Sets)...")
    sas_dir = RAW_DIR / "asap_sas"
    sas_dir.mkdir(parents=True, exist_ok=True)
    
    tsv_path = sas_dir / "train.tsv"
    if tsv_path.exists() and os.path.getsize(tsv_path) > 1000000:
        print(f"  -> ASAP-SAS full dataset already exists: {tsv_path.name} ({os.path.getsize(tsv_path) / (1024*1024):.2f} MB)")
        return
    
    try:
        import kaggle
        print("  -> Downloading ASAP-SAS via Kaggle API (harshdevgoyal/short-answer-scoring)...")
        kaggle.api.dataset_download_files("harshdevgoyal/short-answer-scoring", path=str(sas_dir), unzip=True)
        print(f"  -> Successfully downloaded & extracted ASAP-SAS full corpus to {sas_dir}")
    except Exception as e:
        print(f"  [!] Kaggle download notice for ASAP-SAS: {e}")

def download_asap_aes_full():
    print("\n[5/5] Downloading ASAP-AES Full Automated Essay Scoring Corpus (8 Prompt Sets)...")
    aes_dir = RAW_DIR / "asap_aes"
    aes_dir.mkdir(parents=True, exist_ok=True)
    
    tsv_path = aes_dir / "training_set_rel3.tsv"
    if tsv_path.exists() and os.path.getsize(tsv_path) > 1000000:
        print(f"  -> ASAP-AES full dataset already exists: {tsv_path.name} ({os.path.getsize(tsv_path) / (1024*1024):.2f} MB)")
        return
    
    try:
        import kaggle
        print("  -> Downloading ASAP-AES via Kaggle API (gamergooo/train-aes)...")
        kaggle.api.dataset_download_files("gamergooo/train-aes", path=str(aes_dir), unzip=True)
        print(f"  -> Successfully downloaded & extracted ASAP-AES full corpus to {aes_dir}")
    except Exception as e:
        print(f"  [!] Kaggle download notice for ASAP-AES: {e}")

def main():
    print("=================================================================")
    print("       FULL-SCALE RAW DATASET DOWNLOADER & LOADER PIPELINE       ")
    print("=================================================================")
    print(f"Target Directory: {RAW_DIR}")

    setup_kaggle_auth()
    download_squad_full()
    download_scientsbank_full()
    download_humaneval_codenet()
    download_asap_sas_full()
    download_asap_aes_full()

    print("\n=================================================================")
    print("Dataset download pipeline finished. All raw datasets ready in data/raw/.")
    print("=================================================================")

if __name__ == "__main__":
    main()
