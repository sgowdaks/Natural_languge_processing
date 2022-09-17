"""
Description:
Date:
Author: 
"""

#the code has been modified and the template is followed only for baseline classifier

import argparse
import numpy as np
from sklearn.metrics import accuracy_score

from tqdm import tqdm
from collections import defaultdict
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class BaselineClassifier():
    """This baseline classifier always predicts positive sentiment."""

    def __init__(self, args):
        self.train_sents, self.train_labels = self.read_data(args.train_file)
        self.val_sents, self.val_labels = self.read_data(args.val_file)

    def read_data(self, filename):
        """Extracts all the sentences and labels from the input file."""
        sents = []
        labels = []
        with open(filename) as f:
            for line in f.readlines():
                line = line.strip().split()
                sents.append(line[1:])
                labels.append(int(line[0]))
        return sents, labels

    def predict(self, corpus):
        """Always predicts a value of 1 given the input corpus."""
        labels = []
        for i in corpus:
            labels.append(1)
        return labels

    def evaluate(self, pred, actual):
        """Evaluates accuracy on training and validation predictions."""
        score = 0
        l = len(pred)
        for i in range(l):
            if pred[i] == actual[i]:
                score = score + 1
        return ((score/l)*100)


class NaiveBayesClassifier(BaselineClassifier):
    """Implements Naive Bayes with unigram features using sklearn."""

    def __init__(self, args):
        super().__init__(args)
        self.token_to_idx = self.extract_unigrams()
        # TODO: Assign a new MultinomialNB() to self.classifier.
        self.classifier = BaselineClassifier
        self.train()

    def extract_unigrams(self):
        """Builds a dictionary of unigrams mapping to indices."""
        # TODO: For each training sentence, assign each new token to an index.
        # vectorizer = CountVectorizer(ngram_range=(1, 1))
        # return vectorizer
        pass


    def compute_features(self, sents):
        """Convert sents to np array of feature vectors X."""
        X = np.zeros((len(sents), len(self.token_to_idx)), dtype=float)
        for index, feat_vec in enumerate(tqdm(X, desc='Load unigram feats')):
            for token in sents[index]:
                token_idx = self.token_to_idx[token]
                X[index][token_idx] += 1
        return X

    def train(self):
        """Trains a Naive Bayes classifier on given input x and labels y."""
        # TODO: Compute features X from self.train_sents.

        # TODO: Convert train_labels to a numpy array of labels y.

        # TODO: Fit the classifier on X and y.

    def predict(self, corpus):
        """Makes predictions with the classifier on computed features."""
        pass


class LogisticRegressionClassifier(NaiveBayesClassifier):

    """Implements logistic regression with unigram features using sklearn."""
    def __init__(self, args):
        BaselineClassifier.__init__(self, args)
        self.token_to_idx = self.extract_unigrams()
        # TODO: Assign a new LogisticRegression() to self.classifier.
        # Hint: You can adjust penalty and C params with command-line args.
        self.train()


class BigramLogisticRegressionClassifier(LogisticRegressionClassifier):
    """Implements logistic regression with unigram and bigram features."""

    def __init__(self, args):
        BaselineClassifier.__init__(self, args)
        self.token_to_idx = self.extract_unigrams()
        self.bigrams_to_idx = self.extract_bigrams()
        # TODO: Assign a new LogisticRegression() to self.classifier.
        # Hint: Be sure to set args.solver.
        self.train()

    def extract_bigrams(self):
        """Builds a dictionary of bigrams mapping to indices."""
        pass

    def compute_features(self, sents):
        """Convert sents to np array of feature vectors X."""
        # TODO: Include both unigram and bigram features.
        pass


def main(args):
    # TODO: Evaluate basline classifier (i.e., always predicts positive).
    # Hint: Should see roughtly 50% accuracy.
    BC = BaselineClassifier(args)
    x_test, y_test = BC.read_data(args.val_file)
    x_train, y_train = BC.read_data(args.train_file)
    k = BC.predict(x_test)
    score = BC.evaluate(k, y_train)
    print("for baseline classifier")
    print("accuracy of base classifier ", score)


    # TODO: Evaluate Naive Bayes classifier with unigram features.
    # Hint: Should see over 90% training and 70% testing accuracy.
    model = MultinomialNB()

    new_xtrain = []
    for i in range(len(x_train)):
        new_xtrain.append(" ".join(x_train[i]))
    xtr, xts, ytr, yts = train_test_split(new_xtrain, y_train, test_size=0.05, random_state=42)

    vectorizer = CountVectorizer(ngram_range=(1, 1))

    # print(xtr[:2])
    X_feats_train = vectorizer.fit_transform(xtr)
    # print(X_feats_train[:2])
    # print(vectorizer.get_feature_names_out())

    new_xtest = []
    for i in range(len(x_test)):
        new_xtest.append(" ".join(x_test[i]))

    model.fit(X_feats_train, ytr)
    print("for multinomial classifier")
    # ypred = model.predict(xts)

    score = model.score(vectorizer.transform(xts), yts)
    print("training accuracy ", score)
    score = model.score(vectorizer.transform(new_xtest), y_test)
    print("accuracy of base classifier ", score)



    # TODO: Evaluate logistic regression classifier with unigrams.
    model = LogisticRegression(solver='liblinear')
    model.fit(X_feats_train, ytr)
    score = model.score(vectorizer.transform(xts), yts)
    print("for logistic regression with unigram ")
    print("training accuracy ", score)
    score = model.score(vectorizer.transform(new_xtest), y_test)
    print("accuracy of base classifier ", score)



    # TODO: Evaluate logistic regression classifier with unigrams + bigrams.
    print("for logistic regression with uni and bigram")
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X_feats_train = vectorizer.fit_transform(xtr)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_feats_train, ytr)
    score = model.score(vectorizer.transform(xts), yts)
    print("training accuracy ", score)
    score = model.score(vectorizer.transform(new_xtest), y_test)
    print("accuracy of base classifier ", score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='sentiment-data/train.txt')
    parser.add_argument('--val_file', default='sentiment-data/val.txt')
    parser.add_argument('--solver', default='liblinear', help='Optimization algorithm.')
    parser.add_argument('--penalty', default='l2', help='Regularization for logistic regression.')
    parser.add_argument('--C', type=float, default=1.0, help='Inverse of regularization strength for logistic regression.')

    args = parser.parse_args()
    main(args)
