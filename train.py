import math
from pathlib import Path
import torch
import torch.nn as nn
import wandb
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity
from hisa_model import HiSAGPT
from ds import get_tokenizer, get_dataloaders

config = {
    "batch_size": 250,
    "num_epochs": 30,
    "d_model": 324,
    "nhead": 6,
    "num_layers": 6,
    "dropout": 0.1,
    "lr": 1e-4,
    "vocab_size": 32000,
    "seq_length": 512,
    "sparsity": 0.9,
    "use_sparse": False,
    "model_name": "HiSAGPT_PoC",
    "dataset_fraction": 0.1,  # Use only 10% of the dataset
}


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
    accumulation_steps=2,
):
    scaler = GradScaler()
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_words = 0
        optimizer.zero_grad()
        for i, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        ):
            inputs = batch.to(device)
            targets = inputs.clone()
            targets[:, :-1] = inputs[:, 1:]
            targets[:, -1] = tokenizer.pad_id()
            with autocast():
                output = model(inputs)
                loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                wandb.log({"grad_norm": grad_norm})
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            total_loss += loss.item() * accumulation_steps * targets.numel()
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
            torch.save(model.state_dict(), f"best_{model.__class__.__name__}_model.pth")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer, actual_vocab_size = get_tokenizer(config["vocab_size"])

    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer,
        config["batch_size"],
        config["seq_length"],
        dataset_fraction=config["dataset_fraction"],
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

    model = HiSAGPT(
        actual_vocab_size,
        config["d_model"],
        config["nhead"],
        config["num_layers"],
        config["dropout"],
        use_sparse=config["use_sparse"],
        sparsity=config["sparsity"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {params}")
    wandb.init(project="hisa_gpt_poc", name="ablated", config=config)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id())

    total_steps = config["num_epochs"] * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)

    try:
        train(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion,
            scheduler,
            device,
            config["num_epochs"],
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
