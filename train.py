import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from config import DEVICE, MAX_GRAD_NORM


def train_epoch(model, data_loader, optimizer, scheduler):
    """Run one full training epoch. Returns (accuracy, avg_loss)."""
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        outputs = model(input_ids, attention_mask, labels)
        loss, logits = outputs.loss, outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels).item()
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    accuracy = correct_predictions / len(data_loader.dataset)
    return accuracy, np.mean(losses)


def evaluate(model, data_loader):
    """Evaluate the model on a validation set. Returns a dict of metrics."""
    model.eval()
    losses = []
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(input_ids, attention_mask, labels)
            _, preds = torch.max(outputs.logits, dim=1)

            losses.append(outputs.loss.item())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        'accuracy':  accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall':    recall_score(all_labels, all_preds, average='weighted'),
        'f1':        f1_score(all_labels, all_preds, average='weighted'),
        'loss':      np.mean(losses)
    }


def predict(model, data_loader):
    """Run inference on a test set. Returns a list of predicted labels."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            outputs = model(input_ids, attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())

    return predictions
