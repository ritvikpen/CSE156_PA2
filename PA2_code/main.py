import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import Encoder, Decoder, FeedForwardClassifier

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

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts

def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(encoder, classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            # Forward pass through encoder
            embeddings, _ = encoder(X)
            outputs = classifier(embeddings.mean(dim=1))
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    # Data for classifier
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch)

    # Initialize Encoder
    encoder = Encoder(tokenizer.vocab_size)
    encoder.to(device)
    total_enc_params = sum(p.numel() for p in encoder.parameters())
    print("Total number of encoder parameters:", total_enc_params)

    # Initialize Decoder
    decoder = Decoder(tokenizer.vocab_size)
    decoder.to(device)
    total_dec_params = sum(p.numel() for p in decoder.parameters())
    print("Total number of decoder parameters:", total_dec_params)

    # Initialize Classifier
    classifier = FeedForwardClassifier(n_input, n_hidden, n_output)
    classifier.to(device)

    # Optimizers
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=learning_rate)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    # Define the joint loss function
    criterion = torch.nn.CrossEntropyLoss()
  
    # LM Training Data
    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # LM HBush Data
    inputfile = "speechesdataset/test_LM_hbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText1 = f.read()
    test_LM_hbush_dataset = LanguageModelingDataset(tokenizer, lmtestText1,  block_size)
    test_LM_hbush_loader = DataLoader(test_LM_hbush_dataset, batch_size=batch_size, shuffle=True)

    # LM Obama Data
    inputfile = "speechesdataset/test_LM_obama.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText2 = f.read()
    test_LM_obama_dataset = LanguageModelingDataset(tokenizer, lmtestText2,  block_size)
    test_LM_obama_loader = DataLoader(test_LM_obama_dataset, batch_size=batch_size, shuffle=True)

    # LM WBush Data
    inputfile = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtestText3 = f.read()
    test_LM_wbush_dataset = LanguageModelingDataset(tokenizer, lmtestText3,  block_size)
    test_LM_wbush_loader = DataLoader(test_LM_wbush_dataset, batch_size=batch_size, shuffle=True)

    # for the classification  task, you will train for a fixed number of epochs like this:
    for epoch in range(epochs_CLS):
        epoch_loss = 0.0

        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
        
            embeddings, _ = encoder(xb)
            outputs = classifier(embeddings.mean(dim=1))
            loss_cls = criterion(outputs, yb)
            loss_enc = torch.tensor(0.0, requires_grad=True).to(device)
            joint_loss = loss_cls + loss_enc

            optimizer_cls.zero_grad()
            optimizer_enc.zero_grad()
            joint_loss.backward()
            optimizer_cls.step()
            optimizer_enc.step()

            epoch_loss += joint_loss.item()

        train_accuracy = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
        print(f"Epoch [{epoch+1}/{epochs_CLS}], Loss: {epoch_loss / len(train_CLS_loader):.6f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
        

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        _, loss = decoder(xb, yb)
        loss = loss.mean()  # Compute the mean loss across tokens
        
        optimizer_dec.zero_grad()
        loss.backward()
        optimizer_dec.step()

        if (i + 1) % eval_interval == 0:
            train_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
            hbush_perplexity = compute_perplexity(decoder, test_LM_hbush_loader, eval_iters)
            obama_perplexity = compute_perplexity(decoder, test_LM_obama_loader, eval_iters)
            wbush_perplexity = compute_perplexity(decoder, test_LM_wbush_loader, eval_iters)
            print(f"Iteration [{i+1}/{max_iters}], Loss: {loss.item():.6f}, Train Perplexity: {train_perplexity:.6f}, HBush Perplexity: {hbush_perplexity:.6f}, Obama Perplexity: {obama_perplexity:.6f}, WBush Perplexity: {wbush_perplexity:.6f}")


if __name__ == "__main__":
    main()

