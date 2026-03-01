import random
import functools

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_cosine_schedule_with_warmup

import config
from dataset import SentimentDataset, collate_fn
from model import CustomBERT
from train import train_epoch, evaluate, predict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(model):
    """AdamW with weight decay applied only to non-bias/LayerNorm parameters."""
    no_decay = ["bias", "LayerNorm.weight"]
    return AdamW([
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": config.WEIGHT_DECAY, "lr": config.LEARNING_RATE},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, "lr": config.LEARNING_RATE},
    ])


def freeze_last_bert_layer(model):
    """Freeze the final transformer layer (layer 11) to preserve top-level representations."""
    for name, param in model.bert.named_parameters():
        if 'encoder.layer.11' in name:
            param.requires_grad = False


def main():
    set_seed(config.SEED)
    print(f"Using device: {config.DEVICE}")

    # --- Load Data ---
    train_df = pd.read_csv(config.TRAIN_PATH)
    val_df   = pd.read_csv(config.VAL_PATH)
    test_df  = pd.read_csv(config.TEST_PATH)

    num_classes = train_df[config.LABEL_COLUMN].nunique()

    # --- Tokenizer ---
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
    pad_collate = functools.partial(collate_fn, pad_token_id=tokenizer.pad_token_id)

    # --- Datasets & DataLoaders ---
    train_dataset = SentimentDataset(train_df[config.TEXT_COLUMN].values, train_df[config.LABEL_COLUMN].values, tokenizer, config.MAX_LEN)
    val_dataset   = SentimentDataset(val_df[config.TEXT_COLUMN].values,   val_df[config.LABEL_COLUMN].values,   tokenizer, config.MAX_LEN)
    test_dataset  = SentimentDataset(test_df[config.TEXT_COLUMN].values,  np.zeros(len(test_df)),               tokenizer, config.MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,  collate_fn=pad_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
    test_loader  = DataLoader(test_dataset,  batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    # --- Model ---
    model = CustomBERT(config.MODEL_NAME, num_labels=num_classes, dropout_rate=config.DROPOUT_RATE, num_dropouts=config.NUM_DROPOUT)
    freeze_last_bert_layer(model)
    model.to(config.DEVICE)

    # --- Optimizer & Scheduler ---
    optimizer = get_optimizer(model)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(config.WARMUP_RATIO * total_steps), total_steps)

    # --- Training Loop ---
    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        print(f'\nEpoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 30)

        train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        print(f'Train  | loss: {train_loss:.4f}  accuracy: {train_acc:.4f}')

        val_metrics = evaluate(model, val_loader)
        print(f'Val    | loss: {val_metrics["loss"]:.4f}  accuracy: {val_metrics["accuracy"]:.4f}  '
              f'f1: {val_metrics["f1"]:.4f}')

        if val_metrics['accuracy'] > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            best_accuracy = val_metrics['accuracy']
            print(f'  ✓ New best model saved (accuracy: {best_accuracy:.4f})')

    # --- Inference on Test Set ---
    print('\nRunning predictions on test set...')
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    predictions = predict(model, test_loader)

    submission = pd.DataFrame({config.ID_COLUMN: test_df[config.ID_COLUMN], config.LABEL_COLUMN: predictions})
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f'Predictions saved to {config.SUBMISSION_PATH}')


if __name__ == '__main__':
    main()
