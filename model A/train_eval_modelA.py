"""
Training & evaluation script for Model A (ResNet-18 + GRU fusion).

Usage (local / Gilbreth):

    python train_eval_modelA.py \
        --img-root /scratch/gilbreth/ichaudha/balanced-raf-db \
        --text-csv emotions.csv \
        --epochs 5 \
        --batch-size 64

Assumptions:
- Image root has subfolders: train/, val/, test/
  each containing 7 emotion subfolders (angry, disgust, fear, happy, neutral, sad, surprise)
- Text CSV has columns: "text", "label"
  (we currently use only "text" and use the image label as the target)
"""

import argparse
import os
from datetime import datetime
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from modelA_fusion.py import ModelA_MultimodalEmotionNet   # adjust if package structure changes


# ---------------------------------------------------------------------------
# Tokenization & vocabulary
# ---------------------------------------------------------------------------

def simple_tokenize(text: str) -> List[str]:
    """Very simple whitespace + lowercase tokenizer."""
    return text.lower().strip().split()


def build_vocab(texts: List[str], min_freq: int = 2, max_size: int = 40000) -> Dict[str, int]:
    """
    Build a word -> index mapping.

    Special tokens:
        0: <pad>
        1: <unk>
    """
    from collections import Counter

    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))

    # Keep only frequent words
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_size:
            break
        vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int = 40) -> torch.Tensor:
    """Convert a string into a padded tensor of token ids."""
    tokens = simple_tokenize(text)
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in tokens][:max_len]
    if len(ids) < max_len:
        ids += [vocab["<pad>"]] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MultimodalEmotionDataset(Dataset):
    """
    Combines:
      - an ImageFolder (Balanced RAF-DB)
      - a dataframe of text rows (emotions.csv)

    For each image, we attach a text snippet (wrapping around the text dataframe
    if needed) and use the *image* label as the target.

    Note: because the Kaggle text emotion labels do not perfectly align with
    Balanced RAF-DB, we only use the text content, not its label.
    """

    def __init__(
        self,
        img_dir: str,
        text_df: pd.DataFrame,
        vocab: Dict[str, int],
        transform: transforms.Compose,
        max_len: int = 40,
    ) -> None:
        self.image_ds = datasets.ImageFolder(img_dir, transform=transform)
        self.text_df = text_df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.image_ds)

    def __getitem__(self, idx: int):
        img, img_label = self.image_ds[idx]
        # wrap around text rows if image count > text rows
        text_row = self.text_df.iloc[idx % len(self.text_df)]
        text_ids = encode_text(text_row["text"], self.vocab, self.max_len)
        return img, text_ids, img_label


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, text_ids, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        text_ids = text_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images, text_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, text_ids, labels in tqdm(loader, desc="Val/Test", leave=False):
            images = images.to(device)
            text_ids = text_ids.to(device)
            labels = labels.to(device)

            logits = model(images, text_ids)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader.dataset)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc


def save_checkpoint(model, optimizer, epoch, ckpt_path: str):
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        },
        ckpt_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-root", type=str, required=True,
                        help="Root folder for Balanced RAF-DB (contains train/val/test).")
    parser.add_argument("--text-csv", type=str, required=True,
                        help="Path to emotions.csv.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-len", type=int, default=40)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logdir", type=str, default="outputs")
    args = parser.parse_args()

    # ----- Paths --------------------------------------------------------------
    train_dir = os.path.join(args.img_root, "train")
    val_dir = os.path.join(args.img_root, "val")
    test_dir = os.path.join(args.img_root, "test")

    # ----- Device -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ----- Load text dataset --------------------------------------------------
    print("[INFO] Loading text CSV...")
    df_text = pd.read_csv(args.text_csv)
    assert "text" in df_text.columns, "CSV must contain a 'text' column."

    # ----- Build vocabulary ---------------------------------------------------
    print("[INFO] Building vocabulary...")
    vocab = build_vocab(df_text["text"].tolist(), min_freq=2, max_size=40000)
    vocab_size = len(vocab)
    print(f"[INFO] Vocab size: {vocab_size}")

    # ----- Image transforms ---------------------------------------------------
    img_tf = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),  # RAF-DB is grayscale
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # ----- Datasets & loaders -------------------------------------------------
    train_ds = MultimodalEmotionDataset(train_dir, df_text, vocab, img_tf, args.max_len)
    val_ds = MultimodalEmotionDataset(val_dir, df_text, vocab, img_tf, args.max_len)
    test_ds = MultimodalEmotionDataset(test_dir, df_text, vocab, img_tf, args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----- Model, loss, optimizer --------------------------------------------
    model = ModelA_MultimodalEmotionNet(
        vocab_size=vocab_size,
        embed_dim=128,
        hidden_dim=512,
        num_classes=7,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ----- Logging directory --------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.logdir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[INFO] Logs & checkpoints will be saved to: {run_dir}")

    # ----- Training loop ------------------------------------------------------
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n[Epoch {epoch}/{args.epochs}]")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"  Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(run_dir, "best_modelA.pt")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"  [INFO] New best model saved to {ckpt_path}")

    # ----- Final test evaluation ---------------------------------------------
    print("\n[INFO] Evaluating best model on test set...")
    # (Optional) reload best checkpoint here if you want.
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.4f}")


if __name__ == "__main__":
    main()
