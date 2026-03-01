# 🤖 Sentiment Analysis with BERT

Fine-tuned BERT model for multi-class text sentiment classification, built with PyTorch and Hugging Face Transformers.

---

## 📌 Overview

This project fine-tunes a `bert-base-uncased` model on a labeled sentiment dataset. It includes custom preprocessing, a multi-dropout regularization technique, dynamic padding, a cosine learning rate scheduler, and full evaluation metrics.

---

## 🗂️ Project Structure

```
├── bert_sentiment.py       # Main training & inference script
├── submission.csv          # Test set predictions (output)
└── best_model_state.bin    # Best model checkpoint (saved during training)
```

---

## ⚙️ Model Architecture

A custom BERT wrapper (`CustomBERT`) built on top of `bert-base-uncased`:

- **Base model**: `BertModel` from Hugging Face
- **Multi-dropout averaging**: 5 parallel dropout layers (rate = 0.3) whose outputs are averaged before the final classification head — a technique that improves generalization and reduces variance
- **Classifier**: Single linear layer → `num_labels` classes
- **Loss function**: Cross-Entropy Loss

---

## 🛠️ Key Features

| Feature | Details |
|---|---|
| **Preprocessing** | Contraction expansion, URL/mention normalization, HTML unescaping, ASCII cleaning |
| **Tokenization** | `BertTokenizer` with dynamic padding via custom `collate_fn` |
| **Optimizer** | `AdamW` with separate weight decay groups (no decay on biases & LayerNorm) |
| **Scheduler** | Cosine schedule with linear warmup (10% of total steps) |
| **Gradient clipping** | Max norm = 1.0 |
| **Reproducibility** | Fixed seeds for `random`, `numpy`, `torch`, and `cuda` |
| **Device support** | Automatic GPU/CPU detection |

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install torch transformers pandas scikit-learn tqdm
```

### 2. Prepare your data

Place your CSV files in the expected paths and update the file paths at the top of the script:

```python
train_df = pd.read_csv("path/to/train_dataset.csv")
val_df   = pd.read_csv("path/to/val_dataset.csv")
test_df  = pd.read_csv("path/to/test_dataset.csv")
```

Each CSV should have columns: `ID`, `Text`, `Label`.

### 3. Train the model

```bash
python bert_sentiment.py
```

Training runs for **3 epochs**, saving the best checkpoint based on validation accuracy. Predictions for the test set are saved to `submission.csv`.

---

## 📊 Evaluation Metrics

The model is evaluated after each epoch using:

- **Accuracy**
- **Precision** (weighted)
- **Recall** (weighted)
- **F1 Score** (weighted)

---

## 🔧 Hyperparameters

```python
MAX_LEN       = 256
BATCH_SIZE    = 32
EPOCHS        = 3
LEARNING_RATE = 2e-5
DROPOUT_RATE  = 0.3
WARMUP_RATIO  = 0.1
```

---

## 📦 Dependencies

- Python 3.8+
- PyTorch
- Hugging Face `transformers`
- `scikit-learn`
- `pandas`, `numpy`, `tqdm`

---

## 📝 Notes

- The final transformer layer (layer 11) is **frozen** during training to preserve high-level BERT representations and reduce overfitting.
- A `LabelSmoothingCrossEntropy` loss class is included as an optional drop-in replacement for standard Cross-Entropy.
- The project was developed for an academic NLP competition on Kaggle.
