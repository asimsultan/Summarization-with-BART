
import torch
from torch.utils.data import Dataset
from transformers import BartTokenizer

class SummarizationDataset(Dataset):
    def __init__(self, data):
        self.input_texts = data['input_text']
        self.labels = data['summary']

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_texts[idx]),
            'attention_mask': torch.tensor([1] * len(self.input_texts[idx])),
            'labels': torch.tensor(self.labels[idx])
        }

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data(tokenizer, dataset, max_length):
    tokenized_inputs = tokenizer(
        dataset['input_text'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    dataset['input_ids'] = tokenized_inputs['input_ids']
    dataset['attention_mask'] = tokenized_inputs['attention_mask']
    dataset['labels'] = tokenizer(
        dataset['summary'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )['input_ids']
    return dataset
