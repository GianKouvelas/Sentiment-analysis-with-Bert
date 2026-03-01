import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from transformers import BertModel

ModelOutput = namedtuple('ModelOutput', ['loss', 'logits'])


class CustomBERT(nn.Module):
    """
    BERT-based classifier with multi-dropout averaging.

    Instead of a single dropout before the classifier, this model applies
    N parallel dropout layers and averages their outputs. This acts as a
    cheap ensemble and improves generalization without extra inference cost.
    """

    def __init__(self, model_name: str, num_labels: int, dropout_rate: float = 0.3, num_dropouts: int = 5):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_dropouts)])
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output

        # Average logits across all dropout heads
        logits = sum(self.classifier(dropout(pooled_output)) for dropout in self.dropouts)
        logits /= len(self.dropouts)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return ModelOutput(loss=loss, logits=logits)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.
    Can be used as a drop-in replacement for nn.CrossEntropyLoss.
    Helps prevent overconfident predictions.
    """

    def __init__(self, eps: float = 0.1):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = F.log_softmax(pred, dim=1)
        nll_loss = F.nll_loss(log_preds, target, reduction='none')
        smooth_loss = -log_preds.sum(dim=1) / n_classes
        loss = (1 - self.eps) * nll_loss + self.eps * smooth_loss
        return loss.mean()
