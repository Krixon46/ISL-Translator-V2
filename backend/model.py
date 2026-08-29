import torch
import torch.nn as nn


class SignBiLSTM(nn.Module):

    def __init__(
        self,
        input_size=126,
        hidden_size=128,
        num_layers=2,
        num_classes=9
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                64,
                num_classes
            )
        )

    def forward(self, x):

        # Expected:
        # [batch, 20, 126]

        lstm_out, _ = self.lstm(x)

        # Final temporal output
        last_output = lstm_out[:, -1, :]

        output = self.classifier(
            last_output
        )

        return output