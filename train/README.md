# Gradewise AI — Model Architectures & Fine-Tuning Guide

This directory contains the training and fine-tuning pipelines for the models used in **Gradewise AI**.

---

## 1. Model Architecture Code Locations

The model backends, token aligners, embedding scorers, and consensus ensembling engines live in `src/`:

| Component | Code File | Purpose & Architecture |
| :--- | :--- | :--- |
| **DeBERTa-v3 Alignment** | [`src/core/segmentation/deberta_backend.py`](../src/core/segmentation/deberta_backend.py) | Token-level cross-attention, extractive evidence span extraction, and NLI inference. |
| **MPNet Dense Semantic Sim** | [`src/core/segmentation/mpnet_backend.py`](../src/core/segmentation/mpnet_backend.py) | `sentence-transformers/all-mpnet-base-v2` dense embedding generation and cosine similarity calculation. |
| **LLM Reasoning Backend** | [`src/core/segmentation/llm_backend.py`](../src/core/segmentation/llm_backend.py) | Multi-prompt holistic evaluation, rubric decomposition, and chain-of-thought verification. |
| **Consensus Ensembling** | [`src/core/segmentation/ensemble.py`](../src/core/segmentation/ensemble.py) | Multi-model consensus voting and score calibration across all model components. |
| **Lexical Grounding** | [`src/utils/text_align.py`](../src/utils/text_align.py) | Math symbol tokenization, lemma alignment, and lexical ground truth calculation. |
| **Confidence Engine** | [`src/core/confidence_engine.py`](../src/core/confidence_engine.py) | Bayesian variance estimation across models for human-in-the-loop routing. |

---

## 2. Fine-Tuning Scripts (`train/`)

### A. Fine-Tuning DeBERTa-v3 on Student Answers
Fine-tunes `microsoft/deberta-v3-base` or `microsoft/deberta-v3-large` directly on the full 17,207 ASAP-SAS responses or SciEntsBank pairs:

```powershell
python train/train_deberta_asap.py --model_name microsoft/deberta-v3-base --essay_set 1 --epochs 4 --batch_size 16 --lr 2e-5
```

- **Objective Function:** Mean Squared Error (MSE) on continuous score scale.
- **Evaluation Metric:** Quadratic Weighted Kappa (QWK) calculated at every epoch.
- **Output:** Saves the checkpoint with the highest QWK to `models/checkpoints/deberta_asap/set_1/`.

---

### B. LoRA Fine-Tuning for Large Language Models
Fine-tunes open-source LLMs (e.g. Llama-3-8B-Instruct, Mistral-7B) using Parameter-Efficient Fine-Tuning (PEFT / QLoRA):

```powershell
python train/train_lora_llm.py --base_model meta-llama/Meta-Llama-3-8B-Instruct --epochs 3
```
