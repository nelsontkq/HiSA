import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import urllib.request
import io

# Set random seed for reproducibility
torch.manual_seed(42)

# URLs for the PTB dataset
PTB_TRAIN_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt"
PTB_VALID_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt"
PTB_TEST_URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt"

def download_and_read(url):
    with urllib.request.urlopen(url) as response:
        return response.read().decode('utf-8').split('\n')

# Download and read the datasets
train_data = download_and_read(PTB_TRAIN_URL)
val_data = download_and_read(PTB_VALID_URL)
test_data = download_and_read(PTB_TEST_URL)

# Define the tokenizer
tokenizer = get_tokenizer('basic_english')

# Helper function to yield tokens from the dataset
def yield_tokens(data):
    for text in data:
        yield tokenizer(text)

# Build vocabulary
vocab = build_vocab_from_iterator(yield_tokens(train_data), specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])

# Numericalize text function
def numericalize_text(text):
    return [vocab[token] for token in tokenizer(text)]

class PTBDataset(Dataset):
    def __init__(self, data):
        self.data = [numericalize_text(text) for text in data if text.strip()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][:-1]), torch.tensor(self.data[idx][1:])

# Collate function for DataLoader
def collate_batch(batch):
    batch_inputs, batch_targets = zip(*batch)
    batch_inputs = torch.nn.utils.rnn.pad_sequence(batch_inputs, padding_value=vocab['<pad>'], batch_first=True)
    batch_targets = torch.nn.utils.rnn.pad_sequence(batch_targets, padding_value=vocab['<pad>'], batch_first=True)
    return batch_inputs, batch_targets

# Create Datasets
train_dataset = PTBDataset(train_data)
val_dataset = PTBDataset(val_data)
test_dataset = PTBDataset(test_data)

# Create DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# Print some information about the dataset
print(f"Vocabulary size: {len(vocab)}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")

# Example of iterating through the data
for inputs, targets in train_loader:
    print("Input shape:", inputs.shape)
    print("Target shape:", targets.shape)
    print("Sample input:", inputs[0][:10])
    print("Sample target:", targets[0][:10])
    break

# Save vocabulary for later use
torch.save(vocab, 'ptb_vocab.pth')
print("Vocabulary saved to 'ptb_vocab.pth'")