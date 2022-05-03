"""
Description:
Author:
Date:
"""

import argparse
import numpy as np
import loader
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report


torch.manual_seed(1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tagger(nn.Module):
    """TODO: Implement a neural network tagger model of your choice."""

    def __init__(self, embed_dim, hidden_dim, vocab_size, num_y):
        super().__init__()

    def forward(self, text):
        pass


def main(args):

    # Load the training data.
    train_sentences = loader.load_sentences(args.train_file, args.lower)
    train_corpus, dics = loader.prepare_dataset(train_sentences, mode='train', lower=args.lower)
    vocab_size = len(dics['word_to_id'])
    train = []
    # print(dics)
    #print(train_corpus)
    tok_to_ix = dics['word_to_id']
    tag_to_ix = dics['tag_to_id']
    ix_to_tag = dics['id_to_tag']
    print(ix_to_tag)
    tok_to_ix["UNK"] = len(tok_to_ix)

    class LSTM(nn.Module):
        def __init__(self, num_words, emb_dim, num_y, hidden_dim=32):
            super().__init__()
            self.emb = nn.Embedding(num_words, emb_dim)
            self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=1, bidirectional=False)
            self.linear = nn.Linear(hidden_dim, num_y)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, text):
            embeds = self.emb(text)
            out, (last_hidden, last_cell) = self.lstm(embeds.view(len(text), 1, -1))
            tag_space = self.linear(out.view(len(text), -1))
            return self.softmax(tag_space)

    emb_dim = 50
    learning_rate = 0.01
    model = LSTM(len(tok_to_ix), emb_dim, len(tag_to_ix))
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.NLLLoss()

    n_epochs = 5
    for epoch in range(n_epochs):
        model.train()
        i = 0
        for sen in train_corpus:
            x = sen['words']
            y = sen['tags']
            # print(x)
            # print(y)
            x_train_tensor = torch.LongTensor(x)
            y_train_tensor = torch.LongTensor(y)
            # print(len(x_train_tensor))
            # print(len(y_train_tensor))
            pred_y = model(x_train_tensor)
            loss = loss_fn(pred_y, y_train_tensor)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i += 1
            # print(i)
        print("\nEpoch:", epoch)
        print("Training loss:", loss.item())

    # TODO: Build the model.

    # TODO: Train the NN model for the specified number of epochs.

    # Load the validation data for testing.
    test_sentences = loader.load_sentences(args.test_file, args.lower)
    test_corpus = loader.prepare_dataset(test_sentences, mode='test',
                                         lower=args.lower, word_to_id=dics['word_to_id'],
                                         tag_to_id=dics['tag_to_id'])

    # TODO: Evaluate the NN model and compare to the HMM baseline.

    x_test = []
    y_test = []
    y_p = []
    for sent in test_corpus:
        x_test.append(sent["str_words"])
        y_test.append(sent["tags"])
    # print(len(x_test))
    # print(len(y_test))
    y_tes = []
    for line in y_test:
        for i in line:
            # print(i)
            y_tes.append(i)


    with torch.no_grad():
        model.eval()
        for sentence in test_corpus:
            x = []
            for tok in sentence["str_words"]:
                if tok in tok_to_ix:
                    x.append(tok_to_ix[tok])
                else:
                    x.append(tok_to_ix["UNK"])
            x_test = torch.LongTensor(x)
            pred_y_test = model(x_test)
            k = [max_ix for max_ix in pred_y_test.argmax(1).data.numpy()]
            for i in k:
                y_p.append(i)

    # print(len(y_p))
    y_tes = y_tes[:len(y_p)]
    # print(y_tes)
    # print(y_p)
    print("accuracy")
    print(accuracy_score(y_tes, y_p))


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='data/eng.train')
    parser.add_argument('--test_file', default='data/eng.val')
    parser.add_argument('--lower', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate for gradient descent.')
    parser.add_argument('--embed_dim', type=int, default=32, help='Default embedding dimension.')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Default hidden layer dimension.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--model', default='lstm', choices=['ff', 'lstm'])

    args = parser.parse_args()
    main(args)
