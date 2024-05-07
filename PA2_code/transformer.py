# add all  your Encoder and Decoder code here
import torch
from torch import nn
from torch import optim


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

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layer)

    def forward(self, src):
        src = self.embedding(src)
        output = self.transformer_encoder(src)
        return output.mean(dim=1)

class FeedforwardClassifier(nn.Module):
    def __init__(self, n_embd, hidden_dim, num_classes):
        super(FeedforwardClassifier, self).__init__()
        self.fc1 = nn.Linear(n_embd, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    