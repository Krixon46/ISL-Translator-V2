import torch
import torch.nn as nn


class SignLSTM(nn.Module):

    def __init__(
        self,
        input_size=126,
        hidden_size=128,
        num_layers=2,
        num_classes=9
    ):

        super().__init__()

        # ====================================================
        # BIDIRECTIONAL LSTM
        # ====================================================

        self.lstm = nn.LSTM(

            input_size=input_size,

            hidden_size=hidden_size,

            num_layers=num_layers,

            batch_first=True,

            bidirectional=True,

            dropout=0.2 if num_layers > 1 else 0
        )

        # ====================================================
        # CLASSIFIER
        #
        # Bidirectional LSTM:
        #
        # 128 forward
        # +
        # 128 backward
        # =
        # 256
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size * 2,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                num_classes
            )
        )


    def forward(self, x):

        # x shape:
        #
        # batch × sequence × features
        #
        # example:
        #
        # 1 × 20 × 126

        output, _ = self.lstm(x)

        # Last timestep
        last_output = output[:, -1, :]

        # Classification
        logits = self.classifier(
            last_output
        )

        return logits