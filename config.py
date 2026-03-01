import torch

# --- Reproducibility ---
SEED = 42

# --- Model ---
MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 256
NUM_DROPOUT = 5
DROPOUT_RATE = 0.3

# --- Training ---
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.001
WARMUP_RATIO = 0.1
MAX_GRAD_NORM = 1.0

# --- Data ---
TEXT_COLUMN = 'Text'
LABEL_COLUMN = 'Label'
ID_COLUMN = 'ID'

TRAIN_PATH = "data/train_dataset.csv"
VAL_PATH = "data/val_dataset.csv"
TEST_PATH = "data/test_dataset.csv"

# --- Output ---
MODEL_SAVE_PATH = "best_model_state.bin"
SUBMISSION_PATH = "submission.csv"

# --- Device ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
