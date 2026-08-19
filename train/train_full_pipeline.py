"""
Full Multi-Domain Training Pipeline for Automated Answer Grading
================================================================
Trains and fine-tunes DeBERTa-v3 across:
1. ASAP-SAS (Prompt Sets 1–10: 17,207 Short Answers)
2. SciEntsBank (SemEval-2013: 10,804 Science Explanations)
3. ASAP-AES (Long-form Persuasive Essays)

Usage:
    # Full multi-domain training on all sets
    python train/train_full_pipeline.py --epochs 3 --batch_size 16

    # Fast validation training across all sets
    python train/train_full_pipeline.py --fast --epochs 2
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
from sklearn.model_selection import train_test_split
from src.utils.metrics import compute_qwk

class GradingDataset(Dataset):
    """PyTorch Dataset for Student Grading Tasks."""
    def __init__(self, texts, scores, tokenizer, max_length=128):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = float(self.scores[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score": torch.tensor(score, dtype=torch.float)
        }

def train_and_eval_set(model, tokenizer, texts, scores, raw_scores, max_score, set_name, args, device):
    """Trains and evaluates DeBERTa on a specific dataset or prompt set."""
    print(f"\n" + "="*65)
    print(f" TRAINING DOMAIN: {set_name.upper()} ({len(texts):,} Samples | Max Score: {max_score})")
    print("="*65)

    if args.fast and len(texts) > 400:
        print(f"  [*] Fast mode enabled: subsampling to 400 representative samples.")
        texts = texts[:400]
        scores = scores[:400]
        raw_scores = raw_scores[:400]

    train_texts, val_texts, train_scores, val_scores = train_test_split(
        texts, scores, test_size=0.2, random_state=42
    )

    train_ds = GradingDataset(train_texts, train_scores, tokenizer)
    val_ds = GradingDataset(val_texts, val_scores, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.MSELoss()

    best_qwk = -1.0
    out_dir = PROJECT_ROOT / args.output_dir / set_name.lower().replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["score"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = torch.nan_to_num(outputs.logits.squeeze(-1), nan=0.5, posinf=1.0, neginf=0.0)
            logits = torch.clamp(logits, 0.0, 1.0)

            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() if not torch.isnan(loss) else 0.0
            pbar.set_postfix({"mse_loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["score"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.squeeze(-1).cpu().numpy()

                unnorm_preds = np.clip(np.round(logits * max_score), 0, max_score)
                unnorm_trues = np.round(targets.cpu().numpy() * max_score)

                preds.extend(unnorm_preds.astype(int).tolist())
                trues.extend(unnorm_trues.astype(int).tolist())

        qwk = compute_qwk(trues, preds)
        elapsed = time.time() - start_t
        print(f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | Train MSE: {avg_loss:.4f} | Validation QWK: {qwk:.4f}")

        if qwk > best_qwk:
            best_qwk = qwk
            print(f"  [+] Saved new optimal checkpoint -> {out_dir}")
            model.save_pretrained(str(out_dir))
            tokenizer.save_pretrained(str(out_dir))

    return best_qwk

def run_full_training():
    parser = argparse.ArgumentParser(description="Full Multi-Domain DeBERTa Training Pipeline")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--fast", action="store_true", help="Fast training mode with representative sampling")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*65)
    print("      GRADEWISE AI — FULL-SCALE MULTI-DOMAIN TRAINING ENGINE     ")
    print("="*65)
    print(f"Compute Device: {device}")
    print(f"Base Model:     {args.model_name}")
    print(f"Batch Size:     {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")

    print(f"\nInitializing base tokenizer and model from Hugging Face...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    results = {}

    # 1. ASAP-SAS (10 Prompt Sets)
    sas_file = PROJECT_ROOT / "data/raw/asap_sas/train.tsv"
    if sas_file.exists():
        sas_df = pd.read_csv(sas_file, sep="\t")
        sets_to_train = [1, 2] if args.fast else sorted(sas_df["EssaySet"].unique().tolist())
        for eset in sets_to_train:
            sub = sas_df[sas_df["EssaySet"] == eset].copy()
            max_s = sub["Score1"].max()
            sub["norm_score"] = sub["Score1"] / max_s
            
            # Fresh head for each prompt set
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name, num_labels=1, problem_type="regression"
            ).to(device)
            
            best_qwk = train_and_eval_set(
                model=model,
                tokenizer=tokenizer,
                texts=sub["EssayText"].tolist(),
                scores=sub["norm_score"].tolist(),
                raw_scores=sub["Score1"].tolist(),
                max_score=max_s,
                set_name=f"ASAP-SAS Prompt Set {eset}",
                args=args,
                device=device
            )
            results[f"ASAP-SAS Set {eset}"] = best_qwk
    else:
        print(f"[!] Warning: {sas_file} not found. Run data/fetch_full_raw_datasets.py first.")

    # 2. SciEntsBank Science Explanation Corpus
    scibank_dir = PROJECT_ROOT / "data/raw/scientsbank_full"
    if scibank_dir.exists():
        from datasets import load_from_disk
        sb = load_from_disk(str(scibank_dir))
        train_split = sb["train"]
        
        texts = [f"Question: {r['question']}\nStudent: {r['student_answer']}" for r in train_split]
        # Binary / 3-way label conversion
        scores = [1.0 if r.get("label", "") in ["correct", "1", 1] else 0.0 for r in train_split]
        
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=1, problem_type="regression"
        ).to(device)
        
        best_qwk = train_and_eval_set(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            scores=scores,
            raw_scores=scores,
            max_score=1.0,
            set_name="SciEntsBank Science",
            args=args,
            device=device
        )
        results["SciEntsBank Science"] = best_qwk

    # 3. ASAP-AES Long-form Essay Scoring Corpus
    aes_file = PROJECT_ROOT / "data/raw/asap_aes/training_set_rel3.tsv"
    if aes_file.exists():
        aes_df = pd.read_csv(aes_file, sep="\t", encoding="latin1")
        sub = aes_df[aes_df["essay_set"] == 1].copy()
        max_s = float(sub["domain1_score"].max())
        sub["norm_score"] = sub["domain1_score"] / max_s

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=1, problem_type="regression"
        ).to(device)

        best_qwk = train_and_eval_set(
            model=model,
            tokenizer=tokenizer,
            texts=sub["essay"].fillna("").tolist(),
            scores=sub["norm_score"].tolist(),
            raw_scores=sub["domain1_score"].tolist(),
            max_score=max_s,
            set_name="ASAP-AES Persuasive Essay",
            args=args,
            device=device
        )
        results["ASAP-AES Essay"] = best_qwk

    # Print Final Summary Matrix
    print("\n" + "="*65)
    print("             MULTI-DOMAIN TRAINING RESULTS SUMMARY              ")
    print("="*65)
    for domain, qwk in results.items():
        print(f" * {domain:35} | Best QWK: {qwk:.4f}")
    print("="*65)
    print(f"All checkpoints successfully saved to: {PROJECT_ROOT / args.output_dir}")

if __name__ == "__main__":
    run_full_training()
