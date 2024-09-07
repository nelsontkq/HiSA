import math
import time
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import HiSAForPTB


def yield_tokens(data_iter):
    tokenizer = get_tokenizer("basic_english")
    for text in data_iter:
        yield tokenizer(text)


def collate_batch(batch, vocab, device):
    batch_inputs = [
        torch.tensor([vocab[token] for token in text[:-1]]) for text in batch
    ]
    batch_targets = [
        torch.tensor([vocab[token] for token in text[1:]]) for text in batch
    ]
    batch_inputs = nn.utils.rnn.pad_sequence(
        batch_inputs, padding_value=vocab["<pad>"], batch_first=True
    )
    batch_targets = nn.utils.rnn.pad_sequence(
        batch_targets, padding_value=vocab["<pad>"], batch_first=True
    )
    return batch_inputs.to(device), batch_targets.to(device)


def calculate_f1(outputs, targets, vocab):
    pred_flat = outputs.argmax(dim=-1).view(-1)
    targets_flat = targets.view(-1)
    mask = targets_flat != vocab["<pad>"]
    pred_flat = pred_flat[mask]
    targets_flat = targets_flat[mask]

    pred_labels = pred_flat.cpu().numpy()
    true_labels = targets_flat.cpu().numpy()

    return f1_score(true_labels, pred_labels, average="weighted")


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(
            0.0,
            float(total_steps - current_step)
            / float(max(1, total_steps - warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


def train(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs,
    vocab,
):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_f1 = 0
        num_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = batch
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            wandb.log({"grad_norm": grad_norm})
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            f1 = calculate_f1(output, targets, vocab)
            total_f1 += f1
            num_batches += 1

        avg_train_loss = total_loss / num_batches
        avg_train_f1 = total_f1 / num_batches
        val_loss, val_f1 = evaluate(model, val_loader, criterion, device, vocab)

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_f1": avg_train_f1,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        print(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train F1: {avg_train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_hisa_model.pth")
            print("Best model saved!")


def evaluate(model, data_loader, criterion, device, vocab):
    model.eval()
    total_loss = 0
    total_f1 = 0
    num_batches = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            output = model(inputs)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            total_loss += loss.item()
            f1 = calculate_f1(output, targets, vocab)
            total_f1 += f1
            num_batches += 1
    return total_loss / num_batches, total_f1 / num_batches


def calculate_perplexity(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_words = 0
    start_time = time.time()

    with torch.no_grad():
        for batch in data_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            total_loss += loss.item() * targets.numel()
            total_words += targets.numel()

    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)

    elapsed_time = time.time() - start_time
    words_per_second = total_words / elapsed_time

    print(f"Perplexity: {perplexity:.3f}")
    print(f"Speed: {words_per_second:.0f} words/sec")

    return perplexity


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    wandb.init(project="hisa-ptb")

    # Hyperparameters
    batch_size = 32
    num_epochs = 25
    d_model = 256
    nhead = 4
    num_layers = 4
    dropout = 0.2
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the dataset
    train_iter = PennTreebank(split="train")
    val_iter = PennTreebank(split="valid")
    test_iter = PennTreebank(split="test")

    vocab = build_vocab_from_iterator(
        yield_tokens(train_iter), specials=["<unk>", "<pad>"]
    )
    vocab.set_default_index(vocab["<unk>"])

    train_loader = DataLoader(
        list(train_iter),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, vocab, device),
    )
    val_loader = DataLoader(
        list(val_iter),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, vocab, device),
    )
    test_loader = DataLoader(
        list(test_iter),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, vocab, device),
    )

    # Initialize the model
    model = HiSAForPTB(len(vocab), d_model, nhead, num_layers, dropout).to(device)

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

    # Learning rate scheduler with warmup
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps for warmup
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    # Log config to wandb
    wandb.config.update(
        {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "params": count_parameters(model),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "warmup_steps": warmup_steps,
            "vocab_size": len(vocab),
        }
    )
    try:
        train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            scheduler,
            device,
            num_epochs,
            vocab,
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    # Load the best model and evaluate on test set
    model.load_state_dict(torch.load("best_hisa_model.pth"))
    test_loss, test_f1 = evaluate(model, test_loader, criterion, device, vocab)
    print(f"Final Test Loss: {test_loss:.4f}, Test F1: {test_f1:.4f}")

    test_perplexity = calculate_perplexity(model, test_loader, criterion, device)
    print(f"Test Perplexity: {test_perplexity:.2f}")
    wandb.log(
        {"test_perplexity": test_perplexity, "test_loss": test_loss, "test_f1": test_f1}
    )

    wandb.finish()


if __name__ == "__main__":
    main()
