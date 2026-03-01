import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from preprocess import preprocess_text


class SentimentDataset(Dataset):
    """PyTorch Dataset for tokenized sentiment data."""

    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess_text(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_fn(batch, pad_token_id: int):
    """
    Dynamic padding collate function.
    Pads sequences in a batch to the length of the longest sequence,
    avoiding unnecessary padding to MAX_LEN on shorter batches.
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels
    }
