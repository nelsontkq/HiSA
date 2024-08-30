import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn

from dataset import MICRDataset, collate_fn
from model import MICRTextRecognitionModel


def train(model, train_loader, val_loader, num_epochs, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            output_lengths = torch.full(
                size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long
            )
            target_lengths = torch.full(
                size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.long
            )

            loss = criterion(
                outputs.transpose(0, 1), labels, output_lengths, target_lengths
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                output_lengths = torch.full(
                    size=(outputs.size(0),),
                    fill_value=outputs.size(1),
                    dtype=torch.long,
                )
                target_lengths = torch.full(
                    size=(labels.size(0),), fill_value=labels.size(1), dtype=torch.long
                )
                loss = criterion(
                    outputs.transpose(0, 1), labels, output_lengths, target_lengths
                )
                val_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}"
        )


def main():
    # Set up data transforms
    characters = "0123456789⑈⑆⑉⑇ "
    max_width = 256
    max_label_length = 65
    transform = transforms.Compose(
        [
            transforms.Resize((32, max_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
        ]
    )

    # Create datasets and dataloaders
    train_dataset = MICRDataset(
        "output/train/labels.csv",
        "output/train",
        transform=transform,
        characters=characters,
        max_width=max_width,
        max_label_length=max_label_length,
    )
    val_dataset = MICRDataset(
        "output/val/labels.csv",
        "output/val",
        transform=transform,
        characters=characters,
        max_width=max_width,
        max_label_length=max_label_length,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MICRTextRecognitionModel(num_chars=len(characters) + 1).to(
        device
    )  # +1 for pad token

    # Train the model
    train(model, train_loader, val_loader, num_epochs=50, device=device)

    # Save the model
    torch.save(model.state_dict(), "micr_text_recognition_model.pth")


if __name__ == "__main__":
    main()
