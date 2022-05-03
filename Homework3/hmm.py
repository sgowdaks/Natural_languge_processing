"""
Description: HMM with Viterbi decoding for named entity recognition.
Author: Dr. Korpusik
Reference: Chen & Narasimhan
Date: 6/29/2020
"""

import argparse
import numpy as np
import loader

from sklearn.metrics import classification_report


class HMM():
    """
    Hidden Markov Model (HMM) for named entity recognition.
    Two options for decoding: greedy or Viterbi search.
    """

    def __init__(self, dics, decode_type):
        self.num_words = len(dics['word_to_id'])
        self.num_tags = len(dics['tag_to_id'])

        # Initialize all start, emission, and transition probabilities to 1.
        self.initial_prob = np.ones([self.num_tags])
        self.transition_prob = np.ones([self.num_tags, self.num_tags])
        self.emission_prob = np.ones([self.num_tags, self.num_words])
        self.decode_type = decode_type

    def train(self, corpus):
        """
        TODO: Trains a bigram HMM model using MLE estimates.
        Updates self.initial_prob, self.transition_prob, & self.emission_prob.

        The corpus is a list of dictionaries of the form:
        {'str_words': str_words,   # List of string words
        'words': words,            # List of word IDs
        'tags': tags}              # List of tag IDs

        Each dict's lists all have the same length as that instance's sentence.

        Hint: You should see 90% accuracy with greedy and 91% with Viterbi.
        """

        #calculating the initial probabilites

        self.initial_prob = np.zeros([self.num_tags])
        # print(self.initial_prob)
        for sentence in corpus:
            sen = sentence['tags']
            # print(sentence["tags"])
            self.initial_prob[sen[0]] += 1
        for i in range(len(self.initial_prob)):
            self.initial_prob[i] = self.initial_prob[i] / len(corpus)
        # print(self.initial_prob)

        #calculation the emission probabilites
        ini = np.zeros([self.num_tags])
        self.transition_prob = np.zeros([self.num_tags, self.num_tags])
        # print(self.transition_prob)
        for sentence in corpus:
             sen = sentence['tags']
             #print(sen)
             for i in range(1, len(sen)):
                 ini[sen[i-1]] += 1
                 self.transition_prob[sen[i-1]][sen[i]] += 1
        # print(self.transition_prob)
        # print(ini)

        for p in self.transition_prob:
            p /= np.sum(p)
        #print(self.transition_prob)

        self.emission_prob = np.zeros([self.num_tags, self.num_words])
        for sentence in corpus:
            for i in range(len(sentence['tags'])):
                #print(sentence['tags'][i])
                self.emission_prob[sentence['tags'][i]][sentence['words'][i]] += 1
        for p in self.emission_prob:
            p /= np.sum(p)

        return


    def greedy_decode(self, sentence):
        """
        TODO: Decode a single sentence in greedy fashion.

        The first step uses initial and emission probabilities per tag.
        Each word after the first uses transition and emission probabilities.

        Return a list of greedily predicted tags.
        """

        tags = []

        init_scores = [self.initial_prob[i] * self.emission_prob[i][sentence[0]] for i in range(self.num_tags)]
        tags.append(np.argmax(init_scores))
        #print(tags)
        for word in sentence[1:]:
            scores = [self.transition_prob[tags[-1]][i] * self.emission_prob[i][word] for i in range(self.num_tags)]
            #print(scores)
            tags.append(np.argmax(scores))
        #print(tags)
        assert len(tags) == len(sentence)
        return tags

    def viterbi_decode(self, sentence):
        """
        Decode a single sentence using the Viterbi algorithm.
        Return a list of tags.
        """
        tags = []

        # TODO (optional)

        assert len(tags) == len(sentence)
        return tags

    def tag(self, sentence):
        """
        Tag a sentence using a trained HMM.
        """
        if self.decode_type == 'viterbi':
            return self.viterbi_decode(sentence)
        else:
            return self.greedy_decode(sentence)


def evaluate(model, test_corpus, dics, args):
    """Predicts test data tags with the trained model, and prints accuracy."""
    y_pred = []
    y_actual = []
    for i, sentence in enumerate(test_corpus):
        tags = model.tag(sentence['words'])
        str_tags = [dics['id_to_tag'][tag] for tag in tags]
        y_pred.extend(tags)
        y_actual.extend(sentence['tags'])

    print(classification_report(y_pred, y_actual))


def main(args):
    # Load the training data.
    train_sentences = loader.load_sentences(args.train_file, args.lower)
    train_corpus, dics = loader.prepare_dataset(train_sentences, mode='train',
                                                lower=args.lower)

    #print(train_corpus)
    # Train the HMM.
    model = HMM(dics, decode_type=args.decode_type)
    model.train(train_corpus)

    # Load the validation data for testing.
    test_sentences = loader.load_sentences(args.test_file, args.lower)
    test_corpus = loader.prepare_dataset(test_sentences, mode='test',
                                         lower=args.lower,
                                         word_to_id=dics['word_to_id'],
                                         tag_to_id=dics['tag_to_id'])

    # Evaluate the model on the validation data.
    evaluate(model, test_corpus, dics, args)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_file', default='data/eng.train')
    parser.add_argument('--test_file', default='data/eng.val')
    parser.add_argument('--lower', action='store_true', help='Whether to make all text lowercase.')
    parser.add_argument('--decode_type', default='greedy', choices=['viterbi', 'greedy'])

    args = parser.parse_args()
    main(args)
