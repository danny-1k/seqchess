import torch
from torch import nn



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, device, dropout=0, max_len=5000):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        self.pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, device=device).unsqueeze(1).float()

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.pe[:, 0::2] = torch.sin(position / (10000 ** (_2i / d_model)))
        self.pe[:, 1::2] = torch.cos(position / (10000 ** (_2i / d_model)))

    def forward(self, x):
        _, seq_len, _ = x.shape
        x = x + self.pe[:seq_len, :]
        x = self.dropout(x)

        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, max_len, num_layers, n_heads, d_model, d_ff, device, dropout=0):
        super().__init__()

        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        self.device = device

        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len, dropout=dropout, device=device)


        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                device=device
            ),
            num_layers=num_layers,
        ).to(self.device)

        self.out = nn.Linear(d_model, vocab_size).to(self.device)

    def forward(self, x, mask=None):
        x = self.embedding(x) * (self.d_model**.5)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        x = x.mean(1)
        x = self.out(x)

        return x


class GRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=2, dropout=0, device="cpu"):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_size, device=device)
        self.encoder = nn.GRU(embed_size, hidden_size, num_layers=num_layers, dropout=dropout, device=device, bidirectional=True, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size)
        ).to(device)
        
    def forward(self, inputs):
        # input of shape (N, T)
        # output of shape (N, 1)

        x = self.embedding(inputs) # N, T, hidden_size
        x, _ = self.encoder(x)
        x = self.out(x[:, -1])
    
        return x


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        pass