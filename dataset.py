import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class MICRDataset(Dataset):
    def __init__(
        self,
        root_dir,
        select_data,
        batch_ratio,
        mode,
        max_length,
        img_height,
        img_width,
        character,
        data=None,
    ):
        self.root_dir = root_dir
        self.mode = mode
        self.max_length = max_length
        self.img_height = img_height
        self.img_width = img_width

        if data is None:
            # Read the labels file
            labels_file = os.path.join(root_dir, "labels.csv")
            self.data = pd.read_csv(labels_file)
        else:
            self.data = data

        # Create a character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate(character)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        # Define image transformations
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("L")  # Convert to grayscale
        image = self.transform(image)

        label = self.data.iloc[idx, 1]
        label = self.encode_label(label)

        return image, label

    def encode_label(self, label):
        """Convert string label to list of indices."""
        return [self.char_to_idx[c] for c in label]

    def decode_label(self, label_indices):
        """Convert list of indices back to string label."""
        return "".join(
            [self.idx_to_char[idx] for idx in label_indices if idx in self.idx_to_char]
        )

    def collate_fn(self, batch):
        images, labels = zip(*batch)

        # Find max label length in this batch
        max_label_length = max(len(label) for label in labels)

        # Pad labels to max_label_length
        padded_labels = [
            label + [0] * (max_label_length - len(label)) for label in labels
        ]

        # Convert to tensor
        images = torch.stack(images, 0)
        labels = torch.LongTensor(padded_labels)

        return images, labels
