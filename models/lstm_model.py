import torch.nn as nn

class Seq2SeqLSTMPredictor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=256, num_layers=2):
        super().__init__()

        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=False
        )

        self.decoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=False
        )

        self.output = nn.Linear(hidden_dim, input_dim)
