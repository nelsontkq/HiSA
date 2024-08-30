import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class MICRDataset(Dataset):
    def __init__(self, root_dir, mode='train', max_length=25, img_height=64, img_width=200):
        """
        Args:
            root_dir (string): Directory with all the images and labels.csv file.
            mode (string): 'train' or 'val' to specify the dataset mode.
            max_length (int): Maximum length of the label sequence.
            img_height (int): Height to resize the images to.
            img_width (int): Width to resize the images to.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.max_length = max_length
        self.img_height = img_height
        self.img_width = img_width

        # Read the labels file
        labels_file = os.path.join(root_dir, 'labels.csv')
        self.data = pd.read_csv(labels_file)

        # Create a character to index mapping
        self.char_to_idx = {char: idx for idx, char in enumerate('0123456789⑆⑇⑈⑉')}

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('L')  # Convert to grayscale
        image = self.transform(image)

        label = self.data.iloc[idx, 1]
        label = self.encode_label(label)

        return image, label

    def encode_label(self, label):
        """Convert string label to list of indices."""
        return [self.char_to_idx[c] for c in label]

    def decode_label(self, label_indices):
        """Convert list of indices back to string label."""
        idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        return ''.join([idx_to_char[idx] for idx in label_indices])

    def collate_fn(self, batch):
        images, labels = zip(*batch)
        
        # Pad the labels to max_length
        labels = [torch.LongTensor(label + [0] * (self.max_length - len(label))) for label in labels]
        
        # Stack images and labels
        images = torch.stack(images, 0)
        labels = torch.stack(labels, 0)

        return images, labels