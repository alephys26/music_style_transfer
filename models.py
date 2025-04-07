import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=8, ff_dim=512, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.pos_encoding = self._positional_encoding(embed_dim, 500)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_dim, input_dim)

    def _positional_encoding(self, d_model, max_len):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x, style_emb):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        style_emb = style_emb.permute(1, 0, 2)
        x, _ = self.cross_attention(x, style_emb, style_emb)
        x = self.decoder(x, x)
        x = x.permute(1, 0, 2)
        return self.fc_out(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=8, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 128, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.gru = nn.GRU(embed_dim, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.permute(2, 0, 1)
        x, _ = self.self_attention(x, x, x)
        x = x.permute(1, 0, 2)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return torch.sigmoid(x)


