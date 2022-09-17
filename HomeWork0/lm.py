"""
Description:
Date:
Author: 
"""

import math
import argparse
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import logging as log

log.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                level=log.INFO,
                datefmt='%Y-%m-%d %H:%M:%S')


class LanguageModel:
    """Implements a bigram language model with add-alpha smoothing."""

    def __init__(self, args):
        self.alpha = args.alpha
        self.train_tokens = self.tokenize(args.train_file)
        # print(self.train_tokens)
        self.val_tokens = self.tokenize(args.val_file)

        # Use only the specified fraction of training data.
        num_samples = int(args.train_fraction * len(self.train_tokens))
        self.train_tokens = self.train_tokens[: num_samples]
        self.vocab = self.make_vocab(self.train_tokens)
        self.token_to_idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx_to_token = {i: word for i, word in enumerate(self.vocab)}
        self.count = None
        self.bigrams = self.compute_bigrams(self.train_tokens, 0.0001)
        self.number_of_bigrams = None

    def get_indices(self, tokens):
        """Converts each of the string tokens to indices in the vocab."""
        # log.info("making indices")
        return [self.token_to_idx[token] for token in tqdm(tokens, desc="IDX")
                if token in self.token_to_idx]

    def compute_bigrams(self, tokens, alpha):
        """Populates probability values for a 2D np array of all bigrams."""
        k = len(tokens)
        N = len(self.vocab)
        tokens = self.get_indices(tokens)
        if self.count is None:
            log.info(f"going to calc counts; vocab size is {N}")
            counts = np.zeros((N, N), dtype=np.uint)
            for x, y in tqdm(zip(tokens, tokens[1:])):
                counts[x][y] += 1
            self.count = counts
            log.info("counts calcualted")
        probs = np.zeros((N, N), dtype=np.float)
        counts = self.count
        log.info("making probs")
        for x in tqdm(range(N), total=N):
            total = self.vocab[self.idx_to_token[x]]
            for y in range(N):
                count = counts[x][y]
                probs[x][y] = (count + alpha) / (total + (alpha * N))
        log.info(f"Probs are ready for {alpha}")

        perplexity = 0
        for x, y in tqdm(zip(tokens, tokens[1:])):
            perplexity += np.log2(probs[x][y])
            assert not np.isinf(perplexity), f'Oops! we got Infinite perplexity for ' \
                                             f'{x, y, self.idx_to_token[x], self.idx_to_token[y]}'
        perplexity /= N  # we have n-1 bigrams
        k = np.power(2, -(perplexity))
        return probs, k

    def compute_perplexity(self, tokens):
        """Evaluates the LM by calculating perplexity on the given tokens."""
        tokens = self.get_indices(tokens)

        # TODO: Sum up all the bigram log probabilities in the test corpus.

        # TODO: Be sure to divide by the number of tokens, not the vocab size!

        return 0

    def tokenize(self, corpus):
        """Split the given corpus file into tokens on spaces (or with nltk)."""
        log.info(f"Going to tokenize {corpus}")
        with open(corpus) as f:
            lines = f.readlines()
        lines = " ".join(lines)
        k = word_tokenize(lines)
        log.info("tokenization done")
        return k

    def make_vocab(self, train_tokens):
        """Create a vocabulary dictionary that maps tokens to frequencies."""
        log.info("Making vocab")
        d = Counter(train_tokens)
        log.info(f"vocab done; vocab size:  {len(d)}")
        return d

    def plot_vocab(self, vocab: dict[str, int]):
        """Plot words from most to least common with frequency on the y-axis."""
        items = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

        x = [t for t, _ in items]
        y = [f for _, f in items]

        plt.plot(x, y)
        plt.xlabel('words')
        plt.ylabel('frequency')
        plt.title('Frequency of words occurance')


def main(args):
    lm = LanguageModel(args)
    tokenized_words = lm.tokenize(args.train_file)
    test_words = lm.tokenize(args.val_file)
    tokenized_dicti = lm.make_vocab(tokenized_words)
    bigrams = [(s1, s2) for s1, s2 in zip(test_words, test_words[1:])]

    if args.show_plot:
        lm.plot_vocab(tokenized_dicti)

    # TODO: Plot training and validation perplexities as a function of alpha.
    # Hint: Expect ~136 for train and 530 for val when alpha=0.017
    for_train = []
    for_test = []
    alpha = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
    # alpha = [0.1, 1]
    token_inx = lm.token_to_idx
    print("--------perplexity for varying alpha-------")
    for i in alpha:
        print("This is for alpha ", i)
        probs, train_perplex = lm.compute_bigrams(tokenized_words, i)
        print("perplexity for training set is: ",train_perplex)
        for_train.append(train_perplex)
        # print(train_perplex)
        test_perplex = 0
        sum_ = 0
        for j in bigrams:
            if j[0] not in tokenized_words or j[1] not in tokenized_words:
                sum_ = 0 + sum_
            else:
                newl = token_inx[j[0]]
                newm = token_inx[j[1]]
                sum_ = np.log2(probs[newl][newm]) + sum_

        # test_perplex = (np.log2(sum_))
        #print(test_perplex)
        k = np.power(2, -(sum_ / len(test_words)))
        print("perplexity for testing set is: ", k)
        print("\n")
        for_test.append(k)
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set(xlabel="perplexity", ylabel="alpha")
    ax1.set_title('Varying alpha data vs perplexity')
    ax1.plot(for_test, alpha, color='blue', label="Test")
    ax1.plot(for_train, alpha,  color='red', label="train")
    ax1.legend()

    # TODO: Plot train/val perplexities for varying amount of training data.

    alpha = 0.001
    training_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 , 0.9, 1]
    for_train = []
    for_test = []
    print("-----------perplexity for varying traning set------------")
    for i in training_data:
        print("for training data ", i)
        print("\n")
        lm.count = None
        k = int(i * len(tokenized_words))
        train_tokens = tokenized_words[: k]
        # test_words = lm.tokenize(args.val_file)
        # tokenized_dicti = lm.make_vocab(tokenized_words)
        probs, train_perplex = lm.compute_bigrams(train_tokens, alpha)
        print("perplexity for training set: ", train_perplex)
        for_train.append(train_perplex)
        test_perplex = 0
        sum_ = 0
        for j in bigrams:
            if j[0] not in tokenized_words or j[1] not in tokenized_words:
                sum_ = 0 + sum_
            else:
                newl = token_inx[j[0]]
                newm = token_inx[j[1]]
                sum_ = np.log2(probs[newl][newm]) + sum_
        k = np.power(2, -(sum_ / len(test_words)))
        print("perplexity for testing set: ", k)
        print("\n")
        for_test.append(k)

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.set(xlabel="perplxity", ylabel="traning_data")
    ax1.plot(for_test, training_data, color='blue', label="Test")
    ax1.plot(for_train, training_data,  color='red', label="Train")
    ax1.set_title('Varying Training data vs perplexity')
    ax1.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='lm-data/brown-train.txt')
    parser.add_argument('--val_file', default='lm-data/brown-val.txt')
    parser.add_argument('--train_fraction', type=float, default=1.0,
                        help='Specify a fraction of training data to use to train the language model.')
    parser.add_argument('--alpha', type=float, default=0.0001, help='Parameter for add-alpha smoothing.')
    parser.add_argument('--show_plot', type=bool, default=True, help='Whether to display the word frequency plot.')

    args = parser.parse_args()
    main(args)
