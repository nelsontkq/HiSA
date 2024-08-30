import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os


class MICRDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        transform=None,
        characters="0123456789⑈⑆⑉⑇ ",
        max_width=850,
        max_label_length=50,
    ):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.pad_token = "<PAD>"
        self.characters = characters
        self.char_to_idx = {char: idx for idx, char in enumerate(self.characters)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.characters)}
        self.char_to_idx[self.pad_token] = len(self.characters)
        self.idx_to_char[len(self.characters)] = self.pad_token
        self.max_width = max_width
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert("L")  # Convert to grayscale
        label = self.data.iloc[idx, 1]

        # Pad the image
        padded_image = self.pad_image(image)

        if self.transform:
            padded_image = self.transform(padded_image)

        # Encode and pad the label
        label_encoded = self.encode_and_pad_label(label)

        return padded_image, label_encoded

    def pad_image(self, image):
        width, height = image.size
        new_width = self.max_width
        new_height = 43  # Fixed height for MICR lines

        # Create a new white image with the target size
        padded_image = Image.new("L", (new_width, new_height), color="white")

        # Paste the original image into the padded image
        padded_image.paste(image, (0, 0))

        return padded_image

    def encode_and_pad_label(self, label):
        # Encode the label
        encoded = [self.char_to_idx[char] for char in label]

        # Pad or truncate the encoded label
        if len(encoded) < self.max_label_length:
            encoded += [self.char_to_idx[self.pad_token]] * (
                self.max_label_length - len(encoded)
            )
        else:
            encoded = encoded[: self.max_label_length]

        return torch.tensor(encoded, dtype=torch.long)


def collate_fn(batch):
    # Separate images and labels
    images, labels = zip(*batch)

    # Stack images
    images = torch.stack(images, 0)

    # Stack labels
    labels = torch.stack(labels, 0)

    return images, labels
