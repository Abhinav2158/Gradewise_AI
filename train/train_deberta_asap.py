"""
Fine-Tuning Script for DeBERTa-v3 on ASAP-SAS and SciEntsBank Datasets
=======================================================================
This script fine-tunes `microsoft/deberta-v3-base` or `microsoft/deberta-v3-large`
on student responses using MSE regression with Quadratic Weighted Kappa (QWK) evaluation.

Usage:
    python train/train_deberta_asap.py --model_name microsoft/deberta-v3-base --epochs 3 --batch_size 16
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
    from sklearn.model_selection import train_test_split
    from src.utils.metrics import compute_qwk
except ImportError as e:
    print(f"Missing training dependencies: {e}")
    print("Install with: pip install torch transformers scikit-learn accelerate")

class StudentAnswerDataset(Dataset):
    """PyTorch Dataset for Student Answer Scoring."""
    def __init__(self, texts, scores, tokenizer, max_length=256):
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

def load_asap_sas_data(tsv_path, essay_set=1):
    """Loads and prepares ASAP-SAS dataset for training."""
    print(f"Loading raw dataset from {tsv_path} (EssaySet={essay_set})...")
    df = pd.read_csv(tsv_path, sep="\t")
    if essay_set is not None:
        df = df[df["EssaySet"] == essay_set].copy()
    
    # Normalize score between 0 and 1 for stable gradient updates
    max_score = df["Score1"].max()
    df["norm_score"] = df["Score1"] / max_score
    
    texts = df["EssayText"].tolist()
    scores = df["norm_score"].tolist()
    raw_scores = df["Score1"].tolist()
    
    return texts, scores, raw_scores, max_score

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["score"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        
        loss = criterion(logits, targets)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device, max_score):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["score"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1).cpu().numpy()
            
            # Rescale back to original discrete scale
            unnorm_preds = np.clip(np.round(logits * max_score), 0, max_score)
            unnorm_trues = np.round(targets.cpu().numpy() * max_score)
            
            preds.extend(unnorm_preds.astype(int).tolist())
            trues.extend(unnorm_trues.astype(int).tolist())
            
    qwk = compute_qwk(trues, preds)
    return qwk, trues, preds

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3 on Student Answers")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-base", help="Hugging Face Model ID")
    parser.add_argument("--data_path", type=str, default="data/raw/asap_sas/train.tsv", help="Path to raw ASAP-SAS TSV")
    parser.add_argument("--essay_set", type=int, default=1, help="ASAP Prompt Set to train on (1 to 10)")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/deberta_asap", help="Output directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=================================================================")
    print(f"  FINE-TUNING DEBERTA-V3 FOR AUTOMATED SHORT ANSWER SCORING      ")
    print(f"=================================================================")
    print(f"Device:      {device}")
    print(f"Base Model:  {args.model_name}")
    print(f"Batch Size:  {args.batch_size} | LR: {args.lr} | Epochs: {args.epochs}")

    tsv_file = PROJECT_ROOT / args.data_path
    if not tsv_file.exists():
        print(f"Error: Dataset file not found at {tsv_file}.")
        print("Run `python data/fetch_full_raw_datasets.py` first.")
        return

    texts, scores, raw_scores, max_score = load_asap_sas_data(tsv_file, essay_set=args.essay_set)
    print(f"Total samples for Prompt Set {args.essay_set}: {len(texts)} (Max Score: {max_score})")

    # 80/20 Train/Validation Split
    train_texts, val_texts, train_scores, val_scores = train_test_split(
        texts, scores, test_size=0.2, random_state=42
    )

    print(f"Initializing tokenizer and model ({args.model_name})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="regression"
    )
    model.to(device)

    train_ds = StudentAnswerDataset(train_texts, train_scores, tokenizer)
    val_ds = StudentAnswerDataset(val_texts, val_scores, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.MSELoss()

    best_qwk = -1.0
    out_path = PROJECT_ROOT / args.output_dir / f"set_{args.essay_set}"
    out_path.mkdir(parents=True, exist_ok=True)

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        qwk, _, _ = evaluate(model, val_loader, device, max_score)
        print(f"Epoch {epoch}/{args.epochs} | Train MSE Loss: {loss:.4f} | Validation QWK: {qwk:.4f}")

        if qwk > best_qwk:
            best_qwk = qwk
            print(f"  [*] New best QWK ({best_qwk:.4f})! Saving checkpoint to {out_path}...")
            model.save_pretrained(str(out_path))
            tokenizer.save_pretrained(str(out_path))

    print(f"\nTraining Complete! Best Validation QWK: {best_qwk:.4f}")
    print(f"Saved fine-tuned checkpoint to: {out_path}")

if __name__ == "__main__":
    main()
