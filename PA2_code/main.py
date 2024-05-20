import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
import os
import sys

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import Encoder, Decoder, FeedForwardClassifier
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
        loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break

    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    # Select which part of the assignment to run 
    if(len(sys.argv) == 2):
        part = sys.argv[1]
    else:
        print("Please provide the part number as an argument (1 for part 1; 2 for part 2; 3 for both)")
        exit()

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch, shuffle=True)

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

    # Initialize Encoder and Sanity Check
    encoder = Encoder(tokenizer.vocab_size)
    encoder.to(device)
    encoder_checker = Utilities(tokenizer, encoder)

    # Initialize Decoder and Sanity Check
    decoder = Decoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer)
    decoder.to(device)
    decoder_checker = Utilities(tokenizer, decoder, True)

    # Initialize Classifier
    classifier = FeedForwardClassifier(n_input, n_hidden, n_output)
    classifier.to(device)

    # Initialize Optimizers
    optimizer_enc = torch.optim.AdamW(encoder.parameters(), lr=learning_rate)
    optimizer_cls = torch.optim.AdamW(classifier.parameters(), lr=learning_rate)
    optimizer_dec = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

    test_sentence = "Every American who serves joins an unbroken line of heroes that stretches from Lexington to Gettysburg ; from Iwo Jima to Inchon ; from Khe Sanh to Kandahar --"

    if part == "1":
        # Sanity Check for Encoder
        encoder_checker.sanity_check(test_sentence, block_size=32)

        for epoch in range(epochs_CLS):

            for xb, yb in train_CLS_loader:

                xb, yb = xb.to(device), yb.to(device)
            
                # CLS training code here
                optimizer_cls.zero_grad(set_to_none=True)
                optimizer_enc.zero_grad(set_to_none=True)

                embeddings, _ = encoder(xb)
                embeddings = embeddings.mean(dim=1)
                logits = classifier(embeddings)
                loss = F.cross_entropy(logits, yb)
                loss.backward()

                optimizer_cls.step()
                optimizer_enc.step()

            train_acc = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
            test_acc = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
            print(f"Epoch {epoch+1}/{epochs_CLS}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

    elif part == '2':

        decoder_checker.sanity_check(test_sentence, block_size=32)

        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            # LM training code here
            loss, _ = decoder(xb, yb)
            optimizer_cls.zero_grad(set_to_none=True)
            optimizer_dec.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_cls.step()
            optimizer_dec.step()
            if (i + 1) % 100 == 0:
                print(f"Step {i+1} Train Perplexity: {compute_perplexity(decoder, train_LM_loader)}")

        hbush_perp = compute_perplexity(decoder, test_LM_hbush_loader)
        obama_perp = compute_perplexity(decoder, test_LM_obama_loader)
        wbush_perp = compute_perplexity(decoder, test_LM_wbush_loader)

        print('Step 500 Obama Perplexity', obama_perp)
        print('Step 500 H. Bush Perplexity', hbush_perp)
        print('Step 500 W. Bush Perplexity', wbush_perp)
    
    elif part == '3':

        # Sanity Check for Encoder
        encoder_checker.sanity_check(test_sentence, block_size=32)

        for epoch in range(epochs_CLS):

            for xb, yb in train_CLS_loader:

                xb, yb = xb.to(device), yb.to(device)
            
                # CLS training code here
                optimizer_cls.zero_grad(set_to_none=True)
                optimizer_enc.zero_grad(set_to_none=True)

                embeddings, _ = encoder(xb)
                embeddings = embeddings.mean(dim=1)
                logits = classifier(embeddings)
                loss = F.cross_entropy(logits, yb)
                loss.backward()

                optimizer_cls.step()
                optimizer_enc.step()

            train_acc = compute_classifier_accuracy(encoder, classifier, train_CLS_loader)
            test_acc = compute_classifier_accuracy(encoder, classifier, test_CLS_loader)
            print(f"Epoch {epoch+1}/{epochs_CLS}, Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

        decoder_checker.sanity_check(test_sentence, block_size=32)

        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            # LM training code here
            loss, _ = decoder(xb, yb)
            optimizer_cls.zero_grad(set_to_none=True)
            optimizer_dec.zero_grad(set_to_none=True)
            loss.backward()
            optimizer_cls.step()
            optimizer_dec.step()
            if (i + 1) % 100 == 0:
                print(f"Step {i+1} Train Perplexity: {compute_perplexity(decoder, train_LM_loader)}")

        hbush_perp = compute_perplexity(decoder, test_LM_hbush_loader)
        obama_perp = compute_perplexity(decoder, test_LM_obama_loader)
        wbush_perp = compute_perplexity(decoder, test_LM_wbush_loader)

        print('Step 500 Obama Perplexity', obama_perp)
        print('Step 500 H. Bush Perplexity', hbush_perp)
        print('Step 500 W. Bush Perplexity', wbush_perp)

    else:
        print("Invalid part number. Please provide 1, 2 or 3 as an argument")


if __name__ == "__main__":
    main()

