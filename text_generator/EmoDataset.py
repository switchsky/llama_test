import os

from torch.utils.data import Dataset


class EmoDataset(Dataset):
    def __init__(self, positive_dir, negative_dir):
        self.data = []
        self.labels = []

        # Load positive samples
        for filename in os.listdir(positive_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(positive_dir, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    text = "Describe the emotion of this text in one word:" + text
                    self.data.append(text)
                    self.labels.append(1)  # Positive label

        # Load negative samples
        for filename in os.listdir(negative_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(negative_dir, filename), 'r', encoding='utf-8') as file:
                    text = file.read()
                    text = "Describe the emotion of this text in one word:" + text
                    self.data.append(text)
                    self.labels.append(0)  # Negative label

    def __getitem__(self, idx):
        return {'text': self.data[idx], 'label': self.labels[idx]}

    def __len__(self):
        return len(self.data)
