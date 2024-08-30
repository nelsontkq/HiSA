import torch
import torch.nn as nn
import torch.nn.functional as F

class MICRTextRecognitionModel(nn.Module):
    def __init__(self, num_chars):
        super(MICRTextRecognitionModel, self).__init__()
        
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(128, 256, bidirectional=True, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(512, num_chars)

    def forward(self, x):
        # CNN feature extraction
        conv = self.cnn(x)
        batch, c, h, w = conv.size()
        conv = conv.view(batch, c, -1).permute(0, 2, 1)  # (batch, seq_len, features)
        
        # RNN sequence processing
        output, _ = self.rnn(conv)
        
        # Predict character at each time step
        output = self.fc(output)
        
        return output