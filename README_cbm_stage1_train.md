# CBM Stage 1 — X → C (Concept Encoder)
### `cbm_stage1_train.py` — Full Code Walkthrough

---

## What is Stage 1 of a CBM?

A **Concept Bottleneck Model (CBM)** splits prediction into two stages:

```
Stage 1:  X ──────────────► C        (this file)
          image        14 concepts
                                  
Stage 2:  C ──────────────► Y        (next file)
         concepts        diagnosis
```

Stage 1 trains a CNN to look at a chest X-ray image and predict
which of 14 medical findings are present. The output is not a diagnosis —
it is a vector of 14 concept scores that a human can inspect and understand.
Stage 2 will then use those concept scores to make the final diagnosis.

---

## Dataset — CheXpert

CheXpert is a large chest X-ray dataset from Stanford.
Each image has labels for 14 radiological findings (the concepts).

```
Label values in the CSV:
   1.0  → finding is present
   0.0  → finding is absent
  -1.0  → uncertain (radiologist was not sure)
   NaN  → not mentioned in the report
```

We use the **U-Ones policy**: uncertain (-1) is treated as positive (1).
This is a common choice — it is safer in medicine to assume a finding
might be present rather than miss it.

---

## CONFIG block

```python
TRAIN_SAMPLES = 400   # how many training images to use
VALID_SAMPLES = 150   # how many validation images to use
TEST_SAMPLES  = 100   # how many test images to use

IMG_SIZE   = 224      # resize every image to 224x224 pixels
BATCH_SIZE = 16       # process 16 images at a time
NUM_EPOCHS = 15       # pass through the training set 15 times
LR         = 1e-3     # learning rate for Adam optimizer
THRESHOLD  = 0.5      # above 0.5 = predict positive, below = predict negative
```

These are small numbers intentionally — this is a quick sanity-check
experiment to verify the pipeline works before scaling up.

---

## CONCEPT_COLS — the 14 labels

```python
CONCEPT_COLS = [
    "No Finding",                  # nothing abnormal in the image
    "Enlarged Cardiomediastinum",  # widened area between the lungs
    "Cardiomegaly",                # enlarged heart
    "Lung Opacity",                # cloudy area in the lung
    "Lung Lesion",                 # abnormal tissue in the lung
    "Edema",                       # fluid in the lungs
    "Consolidation",               # lung filled with liquid/solid material
    "Pneumonia",                   # lung infection
    "Atelectasis",                 # partial lung collapse
    "Pneumothorax",                # air in the chest cavity (collapsed lung)
    "Pleural Effusion",            # fluid around the lungs
    "Pleural Other",               # other pleural abnormality
    "Fracture",                    # broken bone visible in X-ray
    "Support Devices",             # tubes, pacemakers, catheters visible
]
```

Each image gets a separate binary prediction for every one of these.
This is a **multi-label** problem — a patient can have multiple findings
at the same time (e.g., Edema + Pleural Effusion + Support Devices).

---

## UTILS

### `seed_everything(42)`
Sets the same random seed everywhere so results are reproducible.
Without this, every run would give different results due to random
weight initialization and data shuffling.

### `sample_df(df, n, seed)`
Takes a random subset of n rows from a DataFrame.
Used to create small train/val/test splits for fast experimentation.

### `row_to_concepts(row)`
Converts one CSV row into a PyTorch tensor of 14 float values.

```
CSV row:  NaN, 1.0, -1.0, 0.0, ...
Output:   [0.0, 1.0,  1.0, 0.0, ...]   ← tensor of shape (14,)
```

### `compute_pos_weight(df)`
Calculates how imbalanced each concept is in the training data.

```
Example: "Fracture" has 8 positives and 392 negatives out of 400 samples
  pos_weight = 392 / 8 = 49.0  →  clamped to 20.0 (our max)

This tells the loss function:
  "a missed Fracture should be penalized 20x more than a missed negative"
```

Without this, the model learns to always predict 0 for rare findings
because it gets rewarded by the loss for doing so.

---

## DATASET — `CheXpertConceptDataset`

A PyTorch `Dataset` class. It knows how to:
1. Take a row index
2. Find the image file on disk using the path in the CSV
3. Open it with PIL and convert to RGB (X-rays are grayscale but we
   load as RGB because the CNN expects 3 channels)
4. Apply the image transform (resize, normalize)
5. Return `(image_tensor, concept_tensor)` — one training sample

The `DataLoader` wraps this and handles batching, so the training loop
receives 16 images + 16 concept vectors at a time.

---

## MODEL — `SimpleCNN`

A small convolutional neural network built from scratch.

### Feature extractor (learns visual patterns)

```
Input image:    (3, 224, 224)   ← 3 color channels, 224x224 pixels

Conv2d(3→16)  + ReLU            → (16, 112, 112)   stride=2 halves spatial size
MaxPool2d(2)                    → (16,  56,  56)   halves again
Conv2d(16→32) + ReLU            → (32,  28,  28)   stride=2
Conv2d(32→64) + ReLU            → (64,  14,  14)   stride=2
AdaptiveAvgPool2d((1,1))        → (64,   1,   1)   squeeze to single vector
```

Each Conv2d layer learns to detect increasingly complex visual patterns:
- Layer 1: edges, textures
- Layer 2: shapes, regions
- Layer 3: higher-level structures

### Classifier (maps visual features to concept scores)

```
Flatten    → (64,)         ← 64-dimensional feature vector
Linear(64→64) + ReLU       ← learns combinations of features
Dropout(0.2)               ← randomly zeroes 20% of neurons during training
                              to prevent memorizing the training set
Linear(64→14)              ← one output per concept = 14 logits
```

### Output: logits, not probabilities

The model outputs raw **logits** (unbounded real numbers, e.g. -2.3, 0.8, 1.4).
To get probabilities we apply `sigmoid()`:

```
sigmoid(-2.3) = 0.09  → 9%  chance of this finding
sigmoid( 0.8) = 0.69  → 69% chance
sigmoid( 1.4) = 0.80  → 80% chance
```

We keep logits for the loss function because `BCEWithLogitsLoss` is
numerically more stable than applying sigmoid first then computing BCE.

---

## LOSS FUNCTION — `BCEWithLogitsLoss`

**Binary Cross-Entropy (BCE)** measures how wrong the model's probability
is compared to the true label for each concept.

```
For one concept, one sample:
  if y=1 (positive):  loss = -log(p)      ← penalizes low confidence in positive
  if y=0 (negative):  loss = -log(1-p)    ← penalizes high confidence in positive

Perfect prediction (p=1.0 when y=1):  loss = -log(1.0) = 0
Worst prediction   (p=0.0 when y=1):  loss = -log(0.0) = infinity
```

The total loss is the average over all 14 concepts and all 16 images in the batch.

### Why it stays around 1.05

When the model is uncertain (outputs p≈0.5 for everything):
```
BCE(0.5) = -log(0.5) ≈ 0.693 per concept
```
With pos_weight amplifying rare concepts, the average across 14 concepts
naturally sits around 1.0–1.1 at the start of training. It comes down
slowly as the model becomes more confident and more correct.

### Why weighted BCE

Without weighting, the model discovers it can get a low loss by always
predicting 0 for rare concepts. For example:

```
"Fracture": 8 positives, 192 negatives in 200 samples
If model always predicts 0:
  - Gets 192 right, 8 wrong
  - Accuracy = 96%  ← looks great, but useless clinically
```

`pos_weight` fixes this by making wrong predictions on positive (rare)
samples much more costly.

---

## TRAIN LOOP — `train_one_epoch`

```python
model.train()   ← activates Dropout, BatchNorm in training mode
```

For each batch of 16 images:

```
1. Forward pass:   logits = model(imgs)
                   → CNN processes 16 images → 16×14 logit matrix

2. Compute loss:   loss = criterion(logits, concepts)
                   → compares predictions to ground truth labels

3. Backprop:       optimizer.zero_grad()   ← clear previous gradients
                   loss.backward()         ← compute new gradients
                   optimizer.step()        ← update weights

4. Collect preds:  probs = sigmoid(logits)
                   → saved for metric computation at end of epoch
```

After all batches, metrics are computed on the collected predictions.
Note: these train metrics are approximate because weights changed
during the epoch (each batch saw a slightly different model).

---

## VALIDATE — `validate`

```python
model.eval()        ← disables Dropout (all neurons active)
torch.no_grad()     ← disables gradient tracking (faster, less memory)
```

Same forward pass as training but no weight updates.
All predictions come from the same frozen model → metrics are exact.
**This is the trustworthy number. Train metrics are approximate.**

---

## METRICS — `compute_metrics`

### Accuracy
```
all_preds = (probs >= 0.5).astype(int)   ← threshold at 50%
accuracy  = (preds == labels).mean()     ← fraction correct across ALL cells
                                            shape: (N_samples × 14 concepts)
```
Counts every concept of every sample. A model that always predicts 0
will still show ~65–70% accuracy on this dataset because most concepts
are absent in most images. **Do not trust accuracy alone.**

### AUC-ROC (Area Under the ROC Curve)
Answers: *"Can the model correctly rank a positive sample above a negative?"*

```
0.5  = random chance (coin flip)
0.7  = reasonable
0.8  = good
0.9+ = excellent
```

Computed independently for each concept, then averaged (macro average).
If a concept has only one class in the current split (e.g., all negatives),
AUC cannot be computed → it is skipped with a warning.

### AUPRC (Area Under Precision-Recall Curve)
Similar to AUC-ROC but focuses on how well the model finds positives.
More informative than AUC-ROC when positives are rare (which is the case
for most CheXpert concepts). A random classifier scores ≈ prevalence rate.

### F1 Score
```
Precision = of all predicted positives, how many were actually positive?
Recall    = of all actual positives, how many did we predict as positive?
F1        = 2 × (Precision × Recall) / (Precision + Recall)
```
F1 is low here because the model rarely predicts positive → low recall.

---

## DATA SPLITS

```
train_df_full  ← all rows from train.csv
     │
     ├── test_df   (100 rows, seed=99)   ← carved out FIRST
     └── train_df  (400 rows, seed=42)   ← sampled from REMAINING rows
                                            so test never overlaps train

valid_df  ← 150 rows from valid.csv (separate file, no overlap possible)
```

Why disjoint test: if any test image appeared in training, the model
might have memorized it → test metrics would be artificially inflated.

---

## CHECKPOINT SAVING

```python
# Save: convert each tensor to numpy array, store as .npy file
np.save(save_path, {k: v.cpu().numpy() for k, v in model.state_dict().items()})

# Load: reconstruct tensors from numpy arrays
numpy_dict = np.load(save_path, allow_pickle=True).item()
state_dict = {k: torch.tensor(v) for k, v in numpy_dict.items()}
model.load_state_dict(state_dict)
```

We use numpy instead of `torch.save` because `torch.utils.serialization`
is broken in PyTorch 2.5.1. This workaround is functionally identical —
all model weights are preserved exactly.

The best model is defined as the one with the **lowest validation loss**.
It is saved every time validation loss improves.

---

## TEST EVALUATION OUTPUT

```
  Concept                          Pos  Neg    Acc    AUC
  --------------------------------------------------------
  Lung Opacity                      92  108  0.460  0.465
  Fracture                           8  192  0.955  0.762
```

- **Pos / Neg**: how many positive and negative samples for this concept
  in the test subset. Important context — AUC on 8 positives is not reliable.

- **Acc**: per-concept accuracy. High accuracy on rare concepts is misleading
  (model just predicts 0 every time).

- **AUC**: per-concept AUC-ROC. The honest measure. Below 0.5 means the
  model is actively wrong (predicting negatives as positives and vice versa).

---

## Why results are near random (AUC ≈ 0.5)

This is expected for this experimental setup. The causes in order of impact:

| Cause | Fix |
|-------|-----|
| Only 400 training images | Use full CheXpert (~200k images) |
| CNN trained from scratch | Use pretrained backbone (DenseNet121, ResNet50) |
| Only 15 epochs | More epochs (but diminishing returns without more data) |
| Small model (64 features) | Larger model with pretrained weights |

The purpose of this script is to validate that the full pipeline works
correctly end-to-end before scaling up. All components are functioning
as intended.

---

## File structure

```
cbm_work/
├── cbm_stage1_train.py        ← this script
├── data/
│   └── chexpert/
│       └── CheXpert-v1.0-small/
│           ├── train.csv
│           ├── valid.csv
│           └── train/         ← image folders
└── checkpoints/
    └── chexpert_simple_cnn_concept_encoder.npy   ← saved best model
```
