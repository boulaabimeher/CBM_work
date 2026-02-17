# =========================
# ---- Output is below ----
# =========================

import os
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


# =========================
# CONFIG
# =========================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "chexpert")
DATASET_NAME = "CheXpert-v1.0-small"

TRAIN_CSV = os.path.join(DATA_ROOT, DATASET_NAME, "train.csv")
VALID_CSV = os.path.join(DATA_ROOT, DATASET_NAME, "valid.csv")

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- run small experiment ---
TRAIN_SAMPLES = 400
VALID_SAMPLES = 150
TEST_SAMPLES = 100

IMG_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
LR = 1e-3
NUM_WORKERS = 0  # keep at 0 for CPU — avoids multiprocessing overhead
THRESHOLD = 0.5


# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")


# =========================
# CHEXPERT CONCEPTS (14)
# =========================
CONCEPT_COLS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]


# =========================
# UTILS
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_df(df, n, seed=42):
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def row_to_concepts(row, uncertain_as_positive=True):
    """
    Convert CheXpert labels into a float tensor (multi-label).
    NaN -> 0
    -1  -> 1 (if uncertain_as_positive)
    """
    labels = []
    for col in CONCEPT_COLS:
        val = row[col]
        if pd.isna(val):
            val = 0.0
        elif val == -1.0:
            val = 1.0 if uncertain_as_positive else 0.0
        labels.append(float(val))
    return torch.tensor(labels, dtype=torch.float32)


def compute_pos_weight(df):
    """
    pos_weight[i] = neg_i / pos_i per concept.
    Handles class imbalance in BCEWithLogitsLoss.
    Clamped to [1, 20] to avoid extreme values.
    """
    weights = []
    for col in CONCEPT_COLS:
        pos = (df[col] == 1.0).sum() + (df[col] == -1.0).sum()
        neg = len(df) - pos
        w = neg / max(pos, 1)
        weights.append(float(np.clip(w, 1.0, 20.0)))
    return torch.tensor(weights, dtype=torch.float32)


# =========================
# DATASET
# =========================
class CheXpertConceptDataset(Dataset):
    def __init__(self, df, data_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # CSV path example:
        # CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg
        img_path = os.path.join(self.data_root, row["Path"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        concepts = row_to_concepts(row, uncertain_as_positive=True)
        return img, concepts


# =========================
# SIMPLE CNN (FAST CPU)
# =========================
class SimpleCNN(nn.Module):
    """
    A small CNN for quick experiments on CPU.
    Outputs concept logits for BCEWithLogitsLoss.
    """

    def __init__(self, num_concepts=14):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 64 x 1 x 1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_concepts),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# METRICS
# =========================
def compute_metrics(all_labels, all_probs):
    """
    Computes accuracy, macro AUC-ROC, macro AUPRC, macro F1.
    Skips degenerate columns (only one class present) for AUC.
    """
    C = all_labels.shape[1]
    all_preds = (all_probs >= THRESHOLD).astype(int)

    # Element-wise accuracy across all labels and samples
    accuracy = (all_preds == all_labels).mean()

    auc_list = []
    auprc_list = []
    for c in range(C):
        y_true = all_labels[:, c]
        y_prob = all_probs[:, c]
        if len(np.unique(y_true)) < 2:
            continue
        auc_list.append(roc_auc_score(y_true, y_prob))
        auprc_list.append(average_precision_score(y_true, y_prob))

    macro_auc = float(np.mean(auc_list)) if auc_list else float("nan")
    macro_auprc = float(np.mean(auprc_list)) if auprc_list else float("nan")
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "accuracy": accuracy,
        "macro_auc": macro_auc,
        "macro_auprc": macro_auprc,
        "macro_f1": macro_f1,
    }


# =========================
# TRAIN / VALID
# =========================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for imgs, concepts in loader:
        imgs = imgs.to(DEVICE)
        concepts = concepts.to(DEVICE)

        logits = model(imgs)
        loss = criterion(logits, concepts)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_labels.append(concepts.cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    metrics = compute_metrics(all_labels, all_probs)

    return total_loss / len(loader), metrics


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    for imgs, concepts in loader:
        imgs = imgs.to(DEVICE)
        concepts = concepts.to(DEVICE)

        logits = model(imgs)
        loss = criterion(logits, concepts)

        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_labels.append(concepts.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    metrics = compute_metrics(all_labels, all_probs)

    return total_loss / len(loader), metrics


# =========================
# MAIN
# =========================
def main():
    seed_everything(42)

    print("Device:", DEVICE)
    print("Train CSV:", TRAIN_CSV)
    print("Valid CSV:", VALID_CSV)

    if not os.path.exists(TRAIN_CSV) or not os.path.exists(VALID_CSV):
        raise FileNotFoundError(
            "train.csv or valid.csv not found. Check your data folder."
        )

    # Load CSVs
    train_df_full = pd.read_csv(TRAIN_CSV)
    valid_df_full = pd.read_csv(VALID_CSV)

    # Carve out disjoint test subset from train (different seed)
    test_df = sample_df(train_df_full, TEST_SAMPLES, seed=99)
    train_df = sample_df(
        train_df_full.drop(test_df.index, errors="ignore"),
        TRAIN_SAMPLES,
        seed=42,
    )
    valid_df = sample_df(valid_df_full, VALID_SAMPLES, seed=42)

    print(
        f"Using {len(train_df)} train | {len(valid_df)} valid | {len(test_df)} test samples"
    )

    # Compute pos_weight from training labels
    pos_weight = compute_pos_weight(train_df).to(DEVICE)

    # Transforms (keep simple)
    train_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),  # single int avoids collections.Iterable bug
            transforms.CenterCrop(IMG_SIZE),  # ensures exact 224x224 after resize
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    valid_transform = transforms.Compose(
        [
            transforms.Resize(IMG_SIZE),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Dataset + loaders
    train_dataset = CheXpertConceptDataset(
        train_df, data_root=DATA_ROOT, transform=train_transform
    )
    valid_dataset = CheXpertConceptDataset(
        valid_df, data_root=DATA_ROOT, transform=valid_transform
    )
    test_dataset = CheXpertConceptDataset(
        test_df, data_root=DATA_ROOT, transform=valid_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Model — weighted BCE to handle class imbalance
    model = SimpleCNN(num_concepts=len(CONCEPT_COLS)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_valid_loss = float("inf")
    save_path = os.path.join(CHECKPOINT_DIR, "chexpert_simple_cnn_concept_encoder.npy")

    print("\nTraining Stage-1 CBM: X --> C (small CPU experiment)\n")

    # Header — only loss and accuracy during training
    print(
        f"  {'Epoch':<6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}  {'Best'}"
    )
    print("  " + "-" * 58)

    for epoch in range(NUM_EPOCHS):
        train_loss, train_mets = train_one_epoch(
            model, train_loader, optimizer, criterion
        )
        valid_loss, valid_mets = validate(model, valid_loader, criterion)

        saved = ""
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            np.save(
                save_path, {k: v.cpu().numpy() for k, v in model.state_dict().items()}
            )
            saved = "<-- saved"

        print(
            f"  {epoch + 1:<6} {train_loss:>12.4f} {train_mets['accuracy']:>10.4f}"
            f" {valid_loss:>10.4f} {valid_mets['accuracy']:>9.4f}  {saved}"
        )

    print(f"\n  Best val loss : {best_valid_loss:.4f}")
    print(f"  Checkpoint    : {save_path}")

    # =========================
    # TEST EVALUATION
    # =========================
    print("\n" + "=" * 58)
    print("  TEST SET EVALUATION  (best checkpoint)")
    print("=" * 58)

    numpy_dict = np.load(save_path, allow_pickle=True).item()
    state_dict = {k: torch.tensor(v) for k, v in numpy_dict.items()}
    model.load_state_dict(state_dict)

    test_loss, test_mets = validate(model, test_loader, criterion)

    # --- Overall summary ---
    print(f"\n  Samples evaluated : {len(test_df)}")
    print(f"  Loss              : {test_loss:.4f}")
    print(
        f"  Accuracy          : {test_mets['accuracy']:.4f}  ({test_mets['accuracy'] * 100:.1f}%)"
    )
    print(
        f"  Macro AUC-ROC     : {test_mets['macro_auc']:.4f}  (0.5 = random, 1.0 = perfect)"
    )
    print(f"  Macro AUPRC       : {test_mets['macro_auprc']:.4f}")
    print(f"  Macro F1          : {test_mets['macro_f1']:.4f}")

    # --- Per-concept breakdown ---
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, concepts in test_loader:
            logits = model(imgs.to(DEVICE))
            all_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_labels.append(concepts.numpy())

    all_labels = np.vstack(all_labels)  # (N, 14)
    all_probs = np.vstack(all_probs)  # (N, 14)
    all_preds = (all_probs >= THRESHOLD).astype(int)

    print(f"\n  {'Concept':<32} {'Pos':>4} {'Neg':>4} {'Acc':>6} {'AUC':>6}")
    print("  " + "-" * 58)

    for i, concept in enumerate(CONCEPT_COLS):
        y_true = all_labels[:, i]
        y_pred = all_preds[:, i]
        y_prob = all_probs[:, i]

        pos_count = int(y_true.sum())
        neg_count = int(len(y_true) - pos_count)
        acc = (y_pred == y_true).mean()

        if len(np.unique(y_true)) < 2:
            auc_str = "  n/a"
        else:
            auc_str = f"{roc_auc_score(y_true, y_prob):>6.3f}"

        print(f"  {concept:<32} {pos_count:>4} {neg_count:>4} {acc:>6.3f} {auc_str}")

    print("\n  Columns: Pos/Neg = positive/negative samples in test subset")
    print("           Acc = per-concept accuracy | AUC = per-concept AUC-ROC")


if __name__ == "__main__":
    main()


# --- Output is below ---

# Device: cpu
# Train CSV: /cbm_work/data/chexpert/CheXpert-v1.0-small/train.csv
# Valid CSV: /cbm_work/data/chexpert/CheXpert-v1.0-small/valid.csv
# Using 400 train | 150 valid | 100 test samples

# Training Stage-1 CBM: X --> C (small CPU experiment)

#   Epoch    Train Loss  Train Acc   Val Loss   Val Acc  Best
#   ----------------------------------------------------------
#   1            1.0641     0.5229     1.0444    0.6314  <-- saved
#   2            1.0615     0.5625     1.0452    0.6010
#   3            1.0610     0.5696     1.0441    0.5929  <-- saved
#   4            1.0574     0.5732     1.0337    0.6210  <-- saved
#   5            1.0594     0.5918     1.0542    0.6095
#   6            1.0598     0.5677     1.0557    0.6081
#   7            1.0606     0.5645     1.0405    0.6924
#   8            1.0579     0.5668     1.0529    0.6100
#   9            1.0576     0.5627     1.0474    0.6138
#   10           1.0566     0.5616     1.0485    0.5052
#   11           1.0557     0.5577     1.0411    0.5905
#   12           1.0560     0.5641     1.0414    0.6138
#   13           1.0589     0.5723     1.0512    0.5943
#   14           1.0583     0.5863     1.0512    0.5662
#   15           1.0571     0.6025     1.0452    0.6386

#   Best val loss : 1.0337
#   Checkpoint    : /cbm_work/checkpoints/chexpert_simple_cnn_concept_encoder.npy

# ==========================================================
#   TEST SET EVALUATION  (best checkpoint)
# ==========================================================

#   Samples evaluated : 100
#   Loss              : 1.0201
#   Accuracy          : 0.5714  (57.1%)
#   Macro AUC-ROC     : 0.4781  (0.5 = random, 1.0 = perfect)
#   Macro AUPRC       : 0.2327
#   Macro F1          : 0.1366

#   Concept                           Pos  Neg    Acc    AUC
#   ----------------------------------------------------------
#   No Finding                         12   88  0.860  0.364
#   Enlarged Cardiomediastinum         10   90  0.100  0.287
#   Cardiomegaly                       11   89  0.110  0.403
#   Lung Opacity                       45   55  0.550  0.310
#   Lung Lesion                         8   92  0.530  0.550
#   Edema                              29   71  0.710  0.589
#   Consolidation                      22   78  0.250  0.492
#   Pneumonia                          12   88  0.880  0.452
#   Atelectasis                        21   79  0.220  0.559
#   Pneumothorax                        9   91  0.910  0.654
#   Pleural Effusion                   48   52  0.480  0.328
#   Pleural Other                       1   99  0.990  0.980
#   Fracture                            2   98  0.980  0.184
#   Support Devices                    57   43  0.430  0.542

#   Columns: Pos/Neg = positive/negative samples in test subset
#            Acc = per-concept accuracy | AUC = per-concept AUC-ROC
