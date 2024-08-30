import argparse
import os
import string
import sys
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import wandb
import Levenshtein

from model import Model
from dataset import MICRDataset
from utils import CTCLabelConverter, AttnLabelConverter, Averager

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(opt):
    """dataset preparation"""
    train_dataset = MICRDataset(
        opt.train_data,
        opt.select_data,
        opt.batch_ratio,
        "train",
        opt.batch_max_length,
        opt.imgH,
        opt.imgW,
        opt.character,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=int(opt.workers),
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )

    valid_dataset = MICRDataset(
        opt.valid_data,
        opt.select_data,
        opt.batch_ratio,
        "val",
        opt.batch_max_length,
        opt.imgH,
        opt.imgW,
        opt.character,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=valid_dataset.collate_fn,
        pin_memory=True,
    )

    return train_loader, valid_loader


def train(opt):
    # Initialize wandb
    wandb.init(project="micr-ocr", config=vars(opt))

    # Load data
    train_loader, valid_loader = load_data(opt)

    """ model configuration """
    if "CTC" in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model = Model(opt.input_channel, opt.output_channel, opt.hidden_size, opt.num_class)

    # weight initialization
    for name, param in model.named_parameters():
        if "localization_fc2" in name:
            print(f"Skip {name} as it is already initialized")
            continue
        try:
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.kaiming_normal_(param)
        except Exception as e:
            print(f"Warning: Skipped initialization of {name} due to {e}")

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).to(device)
    model.train()

    """ setup loss """
    if "CTC" in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

    # filtered parameters
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))

    # setup optimizer
    optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=opt.lr_decay_step, gamma=0.1
    )

    """ start training """
    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = -1
    i = 0

    for epoch in range(opt.num_epoch):
        for data in tqdm(train_loader):
            # train part
            image_tensors, labels = data
            image = image_tensors.to(device)
            text, length = converter.encode(
                labels, batch_max_length=opt.batch_max_length
            )
            batch_size = image.size(0)

            if "CTC" in opt.Prediction:
                preds = model(image, text).log_softmax(2)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.permute(1, 0, 2)  # to use CTCLoss format

                # Calculate max_length for this batch
                max_length = max(length).item()

                # Resize text tensor to match the prediction length
                if max_length < preds.size(0):
                    text = text[:, :max_length]
                else:
                    preds = torch.nn.functional.pad(
                        preds, (0, 0, 0, 0, 0, max_length - preds.size(0))
                    )

                cost = criterion(preds, text, preds_size, length)

            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )

            model.zero_grad()
            cost.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()

            # validation part
            if i % opt.valInterval == 0:
                elapsed_time = time.time() - start_time
                print(
                    f"[{i}/{len(train_loader)}] Loss: {cost.item():0.5f} elapsed_time: {elapsed_time:0.5f}"
                )
                start_time = time.time()

                model.eval()
                with torch.no_grad():
                    valid_loss, current_accuracy, current_norm_ED, _, _, _, _, _ = (
                        validation(model, criterion, valid_loader, converter, opt)
                    )
                model.train()

                # keep best accuracy model
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(
                        model.state_dict(), f"{opt.saved_model}/best_accuracy.pth"
                    )
                if current_norm_ED > best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(
                        model.state_dict(), f"{opt.saved_model}/best_norm_ED.pth"
                    )

                # log to wandb
                wandb.log(
                    {
                        "train_loss": cost.item(),
                        "valid_loss": valid_loss,
                        "accuracy": current_accuracy,
                        "norm_ED": current_norm_ED,
                        "epoch": epoch,
                    }
                )

            i += 1

        scheduler.step()

    print("end the training")
    wandb.finish()


def validation(model, criterion, valid_loader, converter, opt):
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        current_accuracy = 0
        current_norm_ED = 0
        valid_pbar = tqdm(valid_loader, desc="Validation")
        for data in valid_pbar:
            image_tensors, labels = data
            image = image_tensors.to(device)
            text, length = converter.encode(
                labels, batch_max_length=opt.batch_max_length
            )
            batch_size = image.size(0)

            if "CTC" in opt.Prediction:
                preds = model(image, text)
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.log_softmax(2).permute(1, 0, 2)
                cost = criterion(preds, text, preds_size, length)

                _, preds = preds.max(2)
                preds = preds.transpose(1, 0).contiguous().view(-1)
            else:
                preds = model(image, text[:, :-1])  # align with Attention.forward
                target = text[:, 1:]  # without [GO] Symbol
                cost = criterion(
                    preds.view(-1, preds.shape[-1]), target.contiguous().view(-1)
                )

                _, preds_index = preds.max(2)

            valid_loss += cost.item()

            # Calculate accuracy and norm_ED
            preds_str = converter.decode(preds.data, preds_size.data)
            for pred, gt in zip(preds_str, labels):
                if pred == gt:
                    current_accuracy += 1
                if len(gt) == 0:
                    current_norm_ED += 1
                else:
                    current_norm_ED += 1 - Levenshtein.distance(pred, gt) / max(
                        len(pred), len(gt)
                    )

    valid_loss /= len(valid_loader)
    current_accuracy /= len(valid_loader.dataset)
    current_norm_ED /= len(valid_loader.dataset)

    return valid_loss, current_accuracy, current_norm_ED


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True, help="path to training dataset")
    parser.add_argument(
        "--valid_data", required=True, help="path to validation dataset"
    )
    parser.add_argument(
        "--select_data", type=str, default="/", help="select training data"
    )
    parser.add_argument(
        "--batch_ratio",
        type=str,
        default="1",
        help="assign ratio for each selected data in the batch",
    )
    parser.add_argument(
        "--Prediction", type=str, default="CTC", help="Prediction stage. CTC|Attn"
    )
    parser.add_argument(
        "--workers", type=int, help="number of data loading workers", default=4
    )
    parser.add_argument("--batch_size", type=int, default=192, help="input batch size")
    parser.add_argument(
        "--num_epoch", type=int, default=6, help="number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=1, help="learning rate, default=1.0 for Adadelta"
    )
    parser.add_argument(
        "--beta1", type=float, default=0.9, help="beta1 for adam. default=0.9"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=5, help="gradient clipping value. default=5"
    )
    parser.add_argument(
        "--lr_decay_step", type=int, default=3, help="learning rate decay step"
    )
    parser.add_argument(
        "--valInterval", type=int, default=2000, help="Interval between each validation"
    )
    parser.add_argument(
        "--saveInterval",
        type=int,
        default=50000,
        help="Interval between each model save",
    )
    parser.add_argument(
        "--batch_max_length", type=int, default=25, help="maximum-label-length"
    )
    parser.add_argument(
        "--imgH", type=int, default=32, help="the height of the input image"
    )
    parser.add_argument(
        "--imgW", type=int, default=100, help="the width of the input image"
    )
    parser.add_argument(
        "--input_channel",
        type=int,
        default=1,
        help="the number of input channel of Feature extractor",
    )
    parser.add_argument(
        "--output_channel",
        type=int,
        default=512,
        help="the number of output channel of Feature extractor",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=256, help="the size of the LSTM hidden state"
    )
    parser.add_argument(
        "--character", type=str, default="0123456789⑆⑇⑈⑉ ", help="character label"
    )
    parser.add_argument(
        "--saved_model",
        default="./saved_models",
        help="Where to save the trained model",
    )
    opt = parser.parse_args()

    if not os.path.exists(opt.saved_model):
        os.makedirs(opt.saved_model)

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    train(opt)
