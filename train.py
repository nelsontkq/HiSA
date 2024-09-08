import math
from pathlib import Path
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from hisa_model import HiSAGPT
from ds import get_tokenizer, get_dataloaders


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
    tokenizer,
):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_words = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs = batch.to(device)
            targets = inputs.clone()
            targets[:, :-1] = inputs[:, 1:]
            targets[:, -1] = tokenizer.pad_id()

            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            wandb.log({"grad_norm": grad_norm})
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * targets.numel()
            total_words += targets.numel()

        avg_train_loss = total_loss / total_words
        train_perplexity = math.exp(avg_train_loss)
        val_loss, val_perplexity = evaluate(
            model, val_loader, criterion, device, tokenizer
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "train_perplexity": train_perplexity,
                "val_loss": val_loss,
                "val_perplexity": val_perplexity,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )

        print(
            f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, Val Loss: {val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), f"best_{ model.__class__.__name__}_model.pth"
            )
            print("Best model saved!")


def evaluate(model, data_loader, criterion, device, tokenizer):
    model.eval()
    total_loss = 0
    total_words = 0
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch.to(device)
            targets = inputs.clone()
            targets[:, :-1] = inputs[:, 1:]
            targets[:, -1] = tokenizer.pad_id()

            output = model(inputs)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
            total_loss += loss.item() * targets.numel()
            total_words += targets.numel()
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def main():
    batch_size = 128
    num_epochs = 30
    d_model = 768
    nhead = 12
    num_layers = 12
    dropout = 0.1
    lr = 5e-5
    vocab_size = 32000
    seq_length = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, actual_vocab_size = get_tokenizer(vocab_size)

    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, batch_size, seq_length
    )

    # Debugging loop
    for batch in train_loader:
        print(
            f"Batch shape: {batch.shape}, Max token ID: {batch.max().item()}, Min token ID: {batch.min().item()}"
        )
        unique_tokens = torch.unique(batch)
        print(f"Number of unique tokens: {len(unique_tokens)}")
        print(f"Sample of unique tokens: {unique_tokens[:10]}")
        break

    model = HiSAGPT(actual_vocab_size, d_model, nhead, num_layers, dropout).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {params}")
    wandb.init(project=f"hisa-wikitext-{ params / 1_000_000:.0f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    wandb.config.update(
        {
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": lr,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "warmup_steps": warmup_steps,
            "vocab_size": vocab_size,
            "seq_length": seq_length,
            "model": model.__class__.__name__,
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
            tokenizer,
        )
    except KeyboardInterrupt:
        print("Training interrupted")
        if Path(f"best_{model.__class__.__name__}_model.pth").exists():
            print("Loading best model")
        else:
            return

    model.load_state_dict(torch.load(f"best_{model.__class__.__name__}_model.pth"))
    test_loss, test_perplexity = evaluate(
        model, test_loader, criterion, device, tokenizer
    )
    print(f"Final Test Loss: {test_loss:.4f}, Perplexity: {test_perplexity:.4f}")
    wandb.log({"test_perplexity": test_perplexity, "test_loss": test_loss})

    wandb.finish()


if __name__ == "__main__":
    main()
