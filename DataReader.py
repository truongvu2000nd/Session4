import math
import random
import numpy as np

MAX_DOC_LENGTH = 500

class DataReader:
    """
    A class used to read the data
    """
    def __init__(self, data_path, batch_size):
        self._batch_size = batch_size   # mini batch size
        self._data = []                 # input data
        self._labels = []               # real labels
        self._sentence_lengths = []

        with open(data_path) as f:
            d_lines = f.readlines()

        for data_id, line in enumerate(d_lines):
            features = line.split('<fff>')
            label, doc_id, sentence_length = int(features[0]), int(features[1]), int(features[2])
            tokens = features[3].split()
            tokens = [int(token) for token in tokens]

            self._data.append(tokens)
            self._labels.append(label)
            self._sentence_lengths.append(sentence_length)

        self._data = np.array(self._data)                           # of shape (number of training examples, vocab_size)
        self._labels = np.array(self._labels)       # of shape (number of training examples, 1)
        self._sentence_lengths = np.array(self._sentence_lengths)

    def random_mini_batches(self, seed):
        """
        Randomly shuffles the data and labels
        :return:
        minibatches: A list of tuples, each tuple is a pair of (mini_batch data, mini_batch labels)
        """
        random.seed(seed)
        mini_batches = []
        m = self._data.shape[0]         # number of training examples

        # number of mini batches of size mini_batch_size in your partitionning
        num_complete_minibatches = math.floor(m / self._batch_size)

        # Step 1: Shuffle data and labels
        indices = list(range(m))
        random.shuffle(indices)
        shuffled_data, shuffled_labels, shuffled_sent_lengths = \
            self._data[indices], self._labels[indices], self._sentence_lengths[indices]

        # Step 2: Partition (shuffled_data, shuffled_labels). Minus the end case
        for k in range(0, num_complete_minibatches):
            mini_batch_data = shuffled_data[k * self._batch_size: k * self._batch_size + self._batch_size, :]
            mini_batch_labels = shuffled_labels[k * self._batch_size: k * self._batch_size + self._batch_size]
            mini_batch_sent_lengths = shuffled_sent_lengths[k * self._batch_size: k * self._batch_size + self._batch_size]
            mini_batch = (mini_batch_data, mini_batch_labels, mini_batch_sent_lengths)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % self._batch_size != 0:
            # size of the final batch must be self._batch_size
            # because data has shape (batch_size, MAX_DOC_LENGTH)
            mini_batch_data = shuffled_data[m - self._batch_size: m, :]
            mini_batch_labels = shuffled_labels[m - self._batch_size: m]
            mini_batch_sent_lengths = shuffled_sent_lengths[m - self._batch_size: m]

            mini_batch = (mini_batch_data, mini_batch_labels, mini_batch_sent_lengths)
            mini_batches.append(mini_batch)

        return mini_batches
