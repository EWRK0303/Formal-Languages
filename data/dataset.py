import torch
from torch.utils.data import Dataset

class FormalLanguageDataset(Dataset):
    def __init__(self, data, vocab={'0': 0, '1': 1}):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_str, y = self.data[idx]
        x_tensor = torch.tensor([self.vocab[c] for c in x_str], dtype=torch.long)
        # y can be a string or a one-hot list
        if isinstance(y, str):
            y_tensor = torch.tensor([self.vocab[c] for c in y], dtype=torch.long)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor
