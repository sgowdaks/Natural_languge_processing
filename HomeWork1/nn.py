"""
Description:
Author:
Date:

Potentially Useful References:
https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
https://towardsdatascience.com/multiclass-text-classification-using-lstm-in-pytorch-eac56baed8df
"""

import argparse
import numpy as np
import pandas as pd
import spacy
import string
import logging as log

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.metrics import mean_squared_error

from tqdm.auto import tqdm
import re
import gensim.downloader as api
from sklearn.metrics import classification_report

log.basicConfig(level=log.INFO)
torch.manual_seed(1)

PAD_IDX = 0


class Net(nn.Module):
    """TODO: Implement your own fully-connected neural network!"""

    def __init__(self, num_words, emb_dim, num_y, embeds=None, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(num_words, emb_dim, padding_idx=PAD_IDX)
        if embeds is not None:
            self.emb.weight = nn.Parameter(torch.Tensor(embeds))
        self.chain = nn.Sequential(
            nn.Linear(emb_dim, 50),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(25, num_y)
        )
        log.info(f"Created model = {self}")

    def forward(self, text, get_probs=False):
        x = self.emb(text)
        x = torch.max(x, dim=1)[0]  # max returns both vals and indices, but we only want values
        x = self.chain(x)
        if get_probs:
            x = F.softmax(x, dim=1)
        return x


class LSTM(nn.Module):
    """Optional: Implement an LSTM."""
    pass


class TweetsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = torch.from_numpy(self.X[idx][0].astype(np.int32))
        return seq, self.y[idx], self.X[idx][1]


def train_model(model, epochs, lr, train_dl, val_dl):
    accuracy = 0
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=lr)
    for i in range(epochs):
        model.train()
        sum_loss = 0.0
        total = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x)
            # print(y_pred)

            loss = F.cross_entropy(y_pred, y)
            if loss == None:
                pass
            else:
                loss.backward()
                optimizer.step()
                sum_loss += loss.item() * y.shape[0]
                total += y.shape[0]
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (
        sum_loss / total, val_loss, val_acc, val_rmse))
        accuracy = accuracy + val_acc
    print("average validation accuracy  ", accuracy / epochs)


def validation_metrics(model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item() * y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1))) * y.shape[0]
    return sum_loss / total, correct / total, sum_rmse / total


def main(args):
    # TODO: Load the  data using Pandas dataframes, as in classifier.py.
    tweets = pd.read_csv("./data/labeled_data.csv")
    print(tweets.shape)
    tweets.head()
    tweets = tweets[['class', 'tweet']]
    tweets.columns = ['class', 'tweet']
    tweets.head()
    tok = spacy.load('en')

    def tokenize(text):
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')
        nopunct = regex.sub(" ", text.lower())
        return [token.text for token in tok.tokenizer(nopunct)]

    counts = Counter()
    for index, row in tweets.iterrows():
        counts.update(tokenize(row['tweet']))
    print("num_words before:", len(counts.keys()))
    for word in list(counts):
        if counts[word] < 2:
            del counts[word]
    print("num_words after:", len(counts.keys()))
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)

    def encode_sentence(text, vocab2index, N=70):
        tokenized = tokenize(text)
        encoded = np.zeros(N, dtype=int)
        enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
        length = min(N, len(enc1))
        encoded[:length] = enc1[:length]
        return encoded, length

    tweets['encoded'] = tweets['tweet'].apply(lambda x: np.array(encode_sentence(x, vocab2index)))
    tweets.head()
    X = list(tweets['encoded'])
    y = list(tweets['class'])
    from sklearn.model_selection import train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
    train_ds = TweetsDataset(X_train, y_train)
    valid_ds = TweetsDataset(X_valid, y_valid)

    batch_size = 5000
    vocab_size = len(words)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(valid_ds, batch_size=batch_size)
    # for i, (x, y, l) in enumerate(train_dl):
    #     if i < 5:
    #         print('x', x, x.shape)
    #         print('y', y, len(y))
    #         print('l', l, len(y))
    #     pass
    # passprint("passed")
    num_classes = 3
    epochs = 15
    lr = 0.1
    emb_dim = 50
    embeds = None
    print("traning.......without embedding matrix")
    model2 = Net(vocab_size, 50, 3)
    train_model(model2, epochs, lr, train_dl, val_dl)

    def get_emb_matrix(pretrained, word_counts, emb_size=50):
        """ Creates embedding matrix from word vectors"""
        vocab_size = len(word_counts) + 2
        vocab_to_idx = {}
        vocab = ["", "UNK"]
        W = np.zeros((vocab_size, emb_size), dtype="float32")
        W[0] = np.zeros(emb_size, dtype='float32')  # adding a vector for padding
        W[1] = np.random.uniform(-0.25, 0.25, emb_size)  # adding a vector for unknown words
        vocab_to_idx["UNK"] = 1
        i = 2
        for word in word_counts:
            if word in word_vecs:
                W[i] = word_vecs[word]
            else:
                W[i] = np.random.uniform(-0.25, 0.25, emb_size)
            vocab_to_idx[word] = i
            vocab.append(word)
            i += 1
        return W, np.array(vocab), vocab_to_idx

    word_vecs = api.load('glove-twitter-25').vectors
    print("training.........with embedding matrix")
    pretrained_weights, vocab, vocab2index = get_emb_matrix(word_vecs, counts)
    model = Net(vocab_size, 50, 50, pretrained_weights)
    train_model(model, epochs, lr, train_dl, val_dl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_file', default='data/labeled_data.p')
    parser.add_argument('--new_text', default='./new_text.txt')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for gradient descent.')
    parser.add_argument('--lowercase', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--pretrained', action='store_true', help='Whether to load pre-trained word embeddings.')
    parser.add_argument('--embed_dim', type=int, default=32, help='Default embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Default hidden layer dimension.')
    parser.add_argument('--batch_size', type=int, default=16, help='Default number of examples per minibatch.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--model', default='ff', choices=['ff', 'lstm'])

    args = parser.parse_args()
    main(args)
