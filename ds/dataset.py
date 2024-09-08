import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import sentencepiece as spm


class WikiTextDataset(Dataset):
    def __init__(self, split, tokenizer, seq_length=256):
        self.data = load_dataset("Salesforce/wikitext", "wikitext-103-v1", split=split)
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.vocab_size = tokenizer.get_piece_size()
        self.unk_token_id = tokenizer.unk_id()
        self.pad_token_id = tokenizer.pad_id()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]["text"]
        tokens = self.tokenizer.encode(line, out_type=int)

        # Truncate or pad the sequence
        if len(tokens) > self.seq_length:
            tokens = tokens[: self.seq_length]
        else:
            tokens = tokens + [self.pad_token_id] * (self.seq_length - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)


def get_tokenizer(vocab_size=32000, max_sentence_length=8384):
    if Path(f"wikitext_sp_{vocab_size}.model").exists():
        sp = spm.SentencePieceProcessor()
        sp.load(f"wikitext_sp_{vocab_size}.model")
    else:
        if not Path("wikitext_train.txt").exists():
            # Train SentencePiece model on WikiText train set
            train_data = load_dataset(
                "Salesforce/wikitext", "wikitext-103-v1", split="train"
            )
            with open("wikitext_train.txt", "w") as f:
                for line in train_data["text"]:
                    if len(line) >= max_sentence_length:
                        # find middle space and split
                        split_idx = line.rfind(" ", 0, max_sentence_length)
                        f.write(line[:split_idx] + "\n")
                        f.write(line[split_idx + 1 :] + "\n")
                    else:
                        f.write(line + "\n")

        spm.SentencePieceTrainer.train(
            input="wikitext_train.txt",
            model_prefix=f"wikitext_sp_{vocab_size}",
            vocab_size=vocab_size,
            unk_id=0,
            pad_id=1,
            bos_id=2,
            eos_id=3,
            max_sentence_length=max_sentence_length,
            input_sentence_size=1165043,
            train_extremely_large_corpus=True,
        )
        sp = spm.SentencePieceProcessor()
        sp.load(f"wikitext_sp_{vocab_size}.model")

    return sp, sp.get_piece_size()


def get_dataloaders(tokenizer, batch_size, seq_length, num_workers=4):
    train_dataset = WikiTextDataset("train", tokenizer, seq_length)
    val_dataset = WikiTextDataset("validation", tokenizer, seq_length)
    test_dataset = WikiTextDataset("test", tokenizer, seq_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
