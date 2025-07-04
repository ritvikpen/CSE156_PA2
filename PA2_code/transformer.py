import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch.nn import functional as F
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from utilities import Utilities

seed = 42
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

# add all  your Encoder and Decoder code here
class Encoder(nn.Module):

    class Block(nn.Module):

        class MultiHeadAttention(nn.Module):

            class Head(nn.Module):
                """ one head of self-attention """

                def __init__(self, head_size):
                    super().__init__()
                    self.key = nn.Linear(n_embd, head_size, bias=False)
                    self.query = nn.Linear(n_embd, head_size, bias=False)
                    self.value = nn.Linear(n_embd, head_size, bias=False)

                def forward(self, x):
                    # input of size (batch, time-step, channels)
                    # output of size (batch, time-step, head size)
                    B,T,C = x.shape
                    k = self.key(x)   # (B,T,hs)
                    q = self.query(x) # (B,T,hs)
                    # compute attention scores ("affinities")
                    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
                    wei = F.softmax(wei, dim=-1) # (B, T, T)
                    # perform the weighted aggregation of the values
                    v = self.value(x) # (B,T,hs)
                    out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
                    return out, wei

            """ multiple heads of self-attention in parallel """

            def __init__(self, num_heads, head_size):
                super().__init__()
                self.heads = nn.ModuleList([self.Head(head_size) for _ in range(num_heads)])
                self.proj = nn.Linear(head_size * num_heads, n_embd)

            def forward(self, x):
                out = []
                attns_maps = []
                for head in self.heads:
                    head_out, attn = head(x)
                    out.append(head_out)
                    attns_maps.append(attn)
                out = torch.cat(out, dim=-1)
                out = self.proj(out)
                return out, attns_maps

        class FeedFoward(nn.Module):
            """ a simple linear layer followed by a non-linearity """

            def __init__(self, n_embd):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.ReLU(),
                    nn.Linear(4 * n_embd, n_embd),
                )

            def forward(self, x):
                return self.net(x)

        """ Transformer block: communication followed by computation """

        def __init__(self, n_embd, n_head):
            # n_embd: embedding dimension, n_head: the number of heads we'd like
            super().__init__()
            head_size = n_embd // n_head
            self.sa = self.MultiHeadAttention(n_head, head_size)
            self.ffwd = self.FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            sa_out, attns = self.sa(self.ln1(x))
            x = x + sa_out
            x = x + self.ffwd(self.ln2(x))
            return x, attns
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[self.Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        # print("DEVICE:", device)
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        attns_map = []
        for block in self.blocks:
            x, attns = block(x)
            attns_map.extend(attns)

        x = self.ln_f(x)
        return x, attns_map  # Return embeddings along with attention scores

class FeedForwardClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class Decoder(nn.Module):

    class Block(nn.Module):

        class MultiHeadAttention(nn.Module):
            class Head(nn.Module):
                """ one head of self-attention """

                def __init__(self, head_size, n_embd, block_size, lm=False):
                    super().__init__()
                    self.key = nn.Linear(n_embd, head_size, bias=False)
                    self.query = nn.Linear(n_embd, head_size, bias=False)
                    self.value = nn.Linear(n_embd, head_size, bias=False)
                    self.lm = lm
                    if self.lm:
                        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

                def forward(self, x):
                    # input of size (batch, time-step, channels)
                    # output of size (batch, time-step, head size)
                    B,T,C = x.shape
                    k = self.key(x)
                    q = self.query(x)
                    # compute attention scores ("affinities")
                    wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
                    if self.lm:
                        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
                    wei = F.softmax(wei, dim=-1) # (B, T, T)
                    # perform the weighted aggregation of the values
                    v = self.value(x) # (B,T,hs)
                    out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
                    return out, wei
            
            """ multiple heads of self-attention in parallel """

            def __init__(self, num_heads, head_size, n_embd, block_size, lm):
                super().__init__()
                self.heads = nn.ModuleList([self.Head(head_size, n_embd, block_size, lm) for _ in range(num_heads)])
                self.proj = nn.Linear(head_size * num_heads, n_embd)

            def forward(self, x):
                out, attn = [], []
                for head in self.heads:
                    head_out, head_wei = head(x)
                    out.append(head_out)
                    attn.extend(head_wei)
                out = torch.cat(out, dim=-1)
                out = self.proj(out)
                return out, attn

        class FeedFoward(nn.Module):
            def __init__(self, n_embd):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_embd, 4 * n_embd),
                    nn.ReLU(),
                    nn.Linear(4 * n_embd, n_embd)
                )

            def forward(self, x):
                return self.net(x)

        """ Transformer block: communication followed by computation """
        
        def __init__(self, n_embd, n_head, block_size, lm=False):
            super().__init__()
            head_size = n_embd // n_head
            self.sa = self.MultiHeadAttention(n_head, head_size, n_embd, block_size, lm)
            self.ffwd = self.FeedFoward(n_embd)
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)

        def forward(self, x):
            sa, attn = self.sa(self.ln1(x))
            x = x + sa
            x = x + self.ffwd(self.ln2(x))
            return x, attn

    def __init__(self, vocab_size, n_embd, block_size, n_head, n_layer):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[self.Block(n_embd, n_head, block_size, lm=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.extend(attn)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            loss = None
            
        return loss, attn_maps