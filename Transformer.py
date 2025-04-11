# transformer_nlp_training.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
import seaborn as sns
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader

# Constants
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'fr'

# Tokenizers
token_transform = {
    SRC_LANGUAGE: get_tokenizer('spacy', language='en_core_web_sm'),
    TGT_LANGUAGE: get_tokenizer('spacy', language='fr_core_news_sm')
}

# Token Iterator
def yield_tokens(data_iter, language):
    for src, tgt in data_iter:
        yield token_transform[language](src if language == SRC_LANGUAGE else tgt)

# Vocabulary
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
vocab_transform = {}
for lang in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[lang] = build_vocab_from_iterator(
        yield_tokens(train_iter, lang), specials=['<unk>', '<pad>', '<bos>', '<eos>']
    )
    vocab_transform[lang].set_default_index(vocab_transform[lang]['<unk>'])

# Special tokens
PAD_IDX = vocab_transform[SRC_LANGUAGE]['<pad>']
BOS_IDX = vocab_transform[SRC_LANGUAGE]['<bos>']
EOS_IDX = vocab_transform[SRC_LANGUAGE]['<eos>']

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, max_len=100):
        super().__init__()
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, emb_size, 2)
        angle_rates = 1 / torch.pow(10000, (i / emb_size))
        angle_rads = pos * angle_rates
        pe = torch.zeros(max_len, emb_size)
        pe[:, 0::2] = torch.sin(angle_rads)
        pe[:, 1::2] = torch.cos(angle_rads)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Embedding
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding.embedding_dim)

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_size // num_heads
        self.qkv_linear = nn.Linear(emb_size, emb_size * 3)
        self.out_linear = nn.Linear(emb_size, emb_size)

    def forward(self, q, k, v, mask=None):
        B, T, E = q.shape
        qkv = self.qkv_linear(q).chunk(3, dim=-1)
        q, k, v = [x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2) for x in qkv]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :], float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, E)
        return self.out_linear(out), attn

# Feedforward
class FeedForward(nn.Module):
    def __init__(self, emb_size, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(emb_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, emb_size)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

# Encoder Layer
class EncoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, hidden_size):
        super().__init__()
        self.attn = MultiHeadAttention(emb_size, num_heads)
        self.ff = FeedForward(emb_size, hidden_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

# Decoder Layer
class DecoderLayer(nn.Module):
    def __init__(self, emb_size, num_heads, hidden_size):
        super().__init__()
        self.self_attn = MultiHeadAttention(emb_size, num_heads)
        self.enc_attn = MultiHeadAttention(emb_size, num_heads)
        self.ff = FeedForward(emb_size, hidden_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        self_attn_out, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self_attn_out)
        enc_attn_out, attn_weights = self.enc_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + enc_attn_out)
        ff_out = self.ff(x)
        return self.norm3(x + ff_out), attn_weights

# Encoder
class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, num_layers, num_heads, hidden_size, max_len=100):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, emb_size)
        self.pos = PositionalEncoding(emb_size, max_len)
        self.layers = nn.ModuleList([EncoderLayer(emb_size, num_heads, hidden_size) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        x = self.embed(x) + self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# Decoder
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, num_layers, num_heads, hidden_size, max_len=100):
        super().__init__()
        self.embed = TokenEmbedding(vocab_size, emb_size)
        self.pos = PositionalEncoding(emb_size, max_len)
        self.layers = nn.ModuleList([DecoderLayer(emb_size, num_heads, hidden_size) for _ in range(num_layers)])

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = self.embed(x) + self.pos(x)
        for layer in self.layers:
            x, attn_weights = layer(x, enc_out, tgt_mask, src_mask)
        return x, attn_weights

# Full Transformer
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb_size, num_layers, num_heads, hidden_size):
        super().__init__()
        self.encoder = Encoder(src_vocab, emb_size, num_layers, num_heads, hidden_size)
        self.decoder = Decoder(tgt_vocab, emb_size, num_layers, num_heads, hidden_size)
        self.output_layer = nn.Linear(emb_size, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        dec_out, attn = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.output_layer(dec_out), attn

# Masks
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def create_mask(src, tgt):
    tgt_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_len).to(tgt.device)
    src_mask = None
    return src_mask, tgt_mask

# Collate
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_tensor = torch.tensor([BOS_IDX] + vocab_transform[SRC_LANGUAGE](token_transform[SRC_LANGUAGE](src_sample)) + [EOS_IDX])
        tgt_tensor = torch.tensor([BOS_IDX] + vocab_transform[TGT_LANGUAGE](token_transform[TGT_LANGUAGE](tgt_sample)) + [EOS_IDX])
        src_batch.append(src_tensor)
        tgt_batch.append(tgt_tensor)
    src_batch = nn.utils.rnn.pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

# Training
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_dataloader = DataLoader(list(train_iter), batch_size=32, shuffle=True, collate_fn=collate_fn)

transformer = Transformer(
    src_vocab=len(vocab_transform[SRC_LANGUAGE]),
    tgt_vocab=len(vocab_transform[TGT_LANGUAGE]),
    emb_size=256,
    num_layers=3,
    num_heads=4,
    hidden_size=512
)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

epochs = 1
for epoch in range(epochs):
    transformer.train()
    total_loss = 0
    for src, tgt in train_dataloader:
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        src_mask, tgt_mask = create_mask(src, tgt_input)

        logits, attn = transformer(src, tgt_input, src_mask, tgt_mask)
        loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

# Attention Visualization
sample_attn = attn[0][0].detach().cpu()
def visualize_attention(attn_matrix):
    sns.heatmap(attn_matrix, cmap="viridis")
    plt.title("Attention Heatmap")
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.show()

visualize_attention(sample_attn)
