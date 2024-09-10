import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import wandb
from model import CCN, TransformerBaseline
from data_prep import prepare_data
from tqdm import tqdm
import optuna
from scipy import stats
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
import gc
import numpy as np

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
CLIP = 0.25
BLOCK_SIZE = 1024
NUM_RUNS = 5
EARLY_STOP_PATIENCE = 3
MAX_LOSS_THRESHOLD = 1000  # Maximum acceptable loss value
MIN_IMPROVEMENT_THRESHOLD = 0.001  # Minimum improvement in loss to continue training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def safe_exp(x):
    try:
        return math.exp(x)
    except OverflowError:
        return float('inf')

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")

    for batch, data in progress_bar:
        data = data.to(device)
        targets = data.clone().detach()
        targets[:, :-1] = data[:, 1:]
        targets[:, -1] = data[:, 0]

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, output.size(-1)), targets.view(-1))
        
        if batch != 0 and (torch.isnan(loss) or loss.item() > MAX_LOSS_THRESHOLD):
            print(f"Training failed: Loss is {loss.item()}")
            return None
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        total_loss += loss.item()
        
        if batch % (len(train_loader) // 10) == 0 and batch > 0:
            cur_loss = total_loss / (len(train_loader) // 10)
            elapsed = time.time() - start_time
            progress_bar.set_postfix({
                "lr": optimizer.param_groups[0]["lr"],
                "ms/batch": elapsed * 1000 / (len(train_loader) // 10),
                "loss": cur_loss,
                "ppl": safe_exp(cur_loss),
            })
            total_loss = 0
            start_time = time.time()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return cur_loss

def evaluate(model, eval_loader, criterion):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data in eval_loader:
            data = data.to(device)
            targets = data.clone().detach()
            targets[:, :-1] = data[:, 1:]
            targets[:, -1] = data[:, 0]

            output = model(data)
            loss = criterion(output.view(-1, output.size(-1)), targets.view(-1)).item()
            total_loss += loss

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return total_loss / len(eval_loader)


def calculate_bleu(model, data_loader, tokenizer):
    model.eval()
    references = []
    hypotheses = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = output.max(2)
            for i in range(len(data)):
                reference = tokenizer.decode(data[i].tolist())
                hypothesis = tokenizer.decode(predicted[i].tolist())
                references.append([reference.split()])
                hypotheses.append(hypothesis.split())

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return corpus_bleu(references, hypotheses)


def objective(trial, model_class, train_loader, val_loader, criterion, tokenizer):
    config = {
        "vocab_size": tokenizer.get_piece_size(),
        "dim": trial.suggest_int("dim", 32, 256),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
    }
    if model_class.__name__ == "CCN":
        config["num_beams"] = trial.suggest_int("num_beams", 4, 12)
    else:
        config["num_heads"] = trial.suggest_int("num_heads", 1, 8)
        config["dim"] = config["dim"] - (config["dim"] % config["num_heads"])
    
    model = model_class(**{k: v for k, v in config.items() if k != "learning_rate"}).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float('inf')
    no_improvement_count = 0
    for epoch in range(10):
        train_loss = train(model, train_loader, criterion, optimizer, epoch)
        if train_loss is None:  # Training failed
            return float('inf')
        
        val_loss = evaluate(model, val_loader, criterion)
        
        if val_loss < best_val_loss * (1 - MIN_IMPROVEMENT_THRESHOLD):
            best_val_loss = val_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= EARLY_STOP_PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break
    wandb.log({"objective": {"val_loss": val_loss, "config":config}})

    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_loss

def tune_hyperparameters(model_class, train_loader, val_loader, criterion, tokenizer):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, model_class, train_loader, val_loader, criterion, tokenizer),
        n_trials=20,
        catch=(RuntimeError, ValueError)  # Catch common PyTorch errors
    )
    return study.best_params

def run_experiment(model_class, params, train_loader, val_loader, test_loader, criterion, tokenizer):
    results = []
    for run in range(NUM_RUNS):
        set_seed(run)
        model_params = {k: v for k, v in params.items() if k != "learning_rate"}
        model = model_class(tokenizer.get_piece_size(), **model_params).to(device)
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        best_val_loss = float("inf")
        train_losses = []
        val_losses = []
        no_improvement_count = 0
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train(model, train_loader, criterion, optimizer, epoch)
            if train_loss is None:  # Training failed
                print(f"Training failed for {model_class.__name__} in run {run}")
                break
            
            val_loss = evaluate(model, val_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if val_loss < best_val_loss * (1 - MIN_IMPROVEMENT_THRESHOLD):
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{model_class.__name__}_model_run{run}.pt")
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            scheduler.step()

            if no_improvement_count >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} for {model_class.__name__} in run {run}")
                break

        if train_loss is not None:  # Only evaluate if training didn't fail
            model.load_state_dict(torch.load(f"{model_class.__name__}_model_run{run}.pt"))
            test_loss = evaluate(model, test_loader, criterion)
            bleu_score = calculate_bleu(model, test_loader, tokenizer)
            results.append((test_loss, bleu_score))

            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(f"{model_class.__name__}_learning_curve_run{run}.png")
            plt.close()

        del model, optimizer, scheduler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results

def main():
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_data(BLOCK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    criterion = nn.CrossEntropyLoss()

    wandb.init(project="ccn-vs-transformer-extended")

    print("Tuning hyperparameters for CCN...")
    ccn_params = tune_hyperparameters(
        CCN, train_loader, val_loader, criterion, tokenizer
    )
    print("Tuning hyperparameters for Transformer...")
    wandb.log({ccn_params})
    transformer_params = tune_hyperparameters(
        TransformerBaseline, train_loader, val_loader, criterion, tokenizer
    )
    wandb.log({transformer_params})

    print("Running experiments for CCN...")
    ccn_results = run_experiment(
        CCN, ccn_params, train_loader, val_loader, test_loader, criterion, tokenizer
    )
    print("Running experiments for Transformer...")
    transformer_results = run_experiment(
        TransformerBaseline,
        transformer_params,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        tokenizer,
    )
    if not ccn_results:
        print("All CCN runs failed. Exiting early.")
        return

    if not transformer_results:
        print("All Transformer runs failed. Exiting early.")
        return
    ccn_losses, ccn_bleu = zip(*ccn_results)
    transformer_losses, transformer_bleu = zip(*transformer_results)

    t_statistic_loss, p_value_loss = stats.ttest_ind(ccn_losses, transformer_losses)
    t_statistic_bleu, p_value_bleu = stats.ttest_ind(ccn_bleu, transformer_bleu)

    print(f"CCN average test loss: {sum(ccn_losses)/len(ccn_losses):.4f}")
    print(
        f"Transformer average test loss: {sum(transformer_losses)/len(transformer_losses):.4f}"
    )
    print(
        f"Loss comparison - T-statistic: {t_statistic_loss:.4f}, p-value: {p_value_loss:.4f}"
    )

    print(f"CCN average BLEU score: {sum(ccn_bleu)/len(ccn_bleu):.4f}")
    print(
        f"Transformer average BLEU score: {sum(transformer_bleu)/len(transformer_bleu):.4f}"
    )
    print(
        f"BLEU comparison - T-statistic: {t_statistic_bleu:.4f}, p-value: {p_value_bleu:.4f}"
    )

    wandb.log(
        {
            "ccn_avg_loss": sum(ccn_losses) / len(ccn_losses),
            "transformer_avg_loss": sum(transformer_losses) / len(transformer_losses),
            "ccn_avg_bleu": sum(ccn_bleu) / len(ccn_bleu),
            "transformer_avg_bleu": sum(transformer_bleu) / len(transformer_bleu),
            "loss_t_statistic": t_statistic_loss,
            "loss_p_value": p_value_loss,
            "bleu_t_statistic": t_statistic_bleu,
            "bleu_p_value": p_value_bleu,
        }
    )

    wandb.finish()

if __name__ == "__main__":
    main()
