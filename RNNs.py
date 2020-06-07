import re
from collections import defaultdict
from os import listdir
from os.path import isfile
from DataReader import DataReader

import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

NUM_CLASSES = 20
MAX_DOC_LENGTH = 500


def gather_20newsgroups_data():
    def collect_data_from(parent_dir, list_newsgroups, word_count=None):
        data = []
        for group_id, newsgroup in enumerate(list_newsgroups):
            label = group_id
            dir_path = parent_dir + newsgroup + '/'
            files = [(filename, dir_path + filename)
                     for filename in listdir(dir_path)
                     if isfile(dir_path + filename)]
            files.sort()

            for filename, file_path in files:
                with open(file_path) as file:
                    text = file.read().lower()
                    words = re.split('\W+', text)
                    if word_count is not None:
                        for word in words:
                            word_count[word] += 1

                    content = ' '.join(words)
                    assert len(content.splitlines()) == 1
                    data.append(str(label) + '<fff>' +
                                filename + '<fff>' + content)
        return data

    word_count = defaultdict(int)
    path = './datasets/20news-bydate/'
    train_dir, test_dir = (path + '20news-bydate-train/', path + '20news-bydate-test/')
    list_newsgroups = [newsgroup for newsgroup in listdir(train_dir)]
    list_newsgroups.sort()
    train_data = collect_data_from(train_dir, list_newsgroups, word_count)

    # collect and generate vocab
    vocab = [word for word, freq
             in zip(word_count.keys(), word_count.values()) if freq > 10]
    vocab.sort()
    with open('./datasets/w2v/vocab-raw.txt', 'w') as f:
        f.write('\n'.join(vocab))

    # collect raw data
    test_data = collect_data_from(test_dir, list_newsgroups)
    with open('./datasets/w2v/20news-train-raw.txt', 'w') as f:
        f.write('\n'.join(train_data))
    with open('./datasets/w2v/20news-test-raw.txt', 'w') as f:
        f.write('\n'.join(test_data))


def encode_data(data_path, vocab_path):
    with open(vocab_path) as f:
        vocab = dict([(word, wordID + 2)
                      for wordID, word in enumerate(f.read().splitlines())])
        vocab["*unknown_ID"] = 0
        vocab["*padding_ID"] = 1

    with open(data_path) as f:
        documents = [(line.split("<fff>")[0], line.split("<fff>")[1], line.split("<fff>")[2])
                     for line in f.read().splitlines()]

    encoded_data = []
    for document in documents:
        label, doc_id, text = document
        words = text.split()[:MAX_DOC_LENGTH]
        sentence_length = len(words)
        encoded_text = []
        for word in words:
            if word in vocab:
                encoded_text.append(str(vocab[word]))
            else:
                encoded_text.append(str(vocab["*unknown_ID"]))

        if len(words) < MAX_DOC_LENGTH:
            num_padding = MAX_DOC_LENGTH - len(words)
            encoded_text.extend(str(vocab["*padding_ID"]) * num_padding)

        encoded_data.append(str(label) + '<fff>' + str(doc_id) + '<fff>' +
                            str(sentence_length) + '<fff>' + ' '.join(encoded_text))

        dir_name = '/'.join(data_path.split('/')[:-1])
        file_name = '-'.join(data_path.split('/')[-1].split('-')[:-1]) + '-encoded.txt'
        with open(dir_name + '/' + file_name, 'w') as f:
            f.write('\n'.join(encoded_data))


class RNN:
    def __init__(self, vocab_size, embedding_size, lstm_size, batch_size):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._lstm_size = lstm_size
        self._batch_size = batch_size

        self._data = tf.placeholder(tf.int32, shape=[batch_size, MAX_DOC_LENGTH])
        self._labels = tf.placeholder(tf.int32, shape=[batch_size, ])
        self._sentence_lengths = tf.placeholder(tf.int32, shape=[batch_size, ])
        # self._final_tokens = tf.placeholder(tf.int32, shape=[batch_size, ])

    def embedding_layers(self, indices):
        # word embedding layer
        # the first hidden layer of the network
        # turn indices into dense vectors of shape (, self._embedding_size )
        pretrained_vectors = []
        pretrained_vectors.append(np.zeros(self._embedding_size))
        np.random.seed(2020)
        for _ in range(self._vocab_size + 1):
            pretrained_vectors.append(np.random.normal(size=self._embedding_size))

        pretrained_vectors = np.array(pretrained_vectors)

        self._embedding_matrix = tf.get_variable(
            name='embedding',
            shape=(self._vocab_size + 2, self._embedding_size),
            initializer=tf.constant_initializer(pretrained_vectors)
        )
        return tf.nn.embedding_lookup(self._embedding_matrix, indices)

    def lstm_layer(self, embeddings):
        # lstm layer
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self._lstm_size)
        zero_state = tf.zeros(shape=(self._batch_size, self._lstm_size))
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(zero_state, zero_state)

        lstm_inputs = tf.unstack(tf.transpose(embeddings, perm=[1, 0, 2]))
        lstm_outputs, last_state = tf.nn.static_rnn(
            cell=lstm_cell,
            inputs=lstm_inputs,
            initial_state=initial_state,
            sequence_length=self._sentence_lengths
        )  # a length-500 list of [num_docs, lstm_size]

        lstm_outputs = tf.unstack(tf.transpose(lstm_outputs, perm=[1, 0, 2]))
        lstm_outputs = tf.concat(lstm_outputs, axis=0)  # [num docs * MAX_SENT_LENGTH, lstm_size]
        # self._mask : [num docs * MAX_SENT_LENGTH, ]
        mask = tf.sequence_mask(
            lengths=self._sentence_lengths,
            maxlen=MAX_DOC_LENGTH,
            dtype=tf.float32
        )
        mask = tf.concat(tf.unstack(mask), axis=0)
        mask = tf.expand_dims(mask, -1)
        lstm_outputs = mask * lstm_outputs
        lstm_outputs_split = tf.split(lstm_outputs, num_or_size_splits=self._batch_size)
        lstm_outputs_sum = tf.reduce_sum(lstm_outputs_split, axis=1)  # [num_docs, lstm_size]
        lstm_outputs_average = lstm_outputs_sum / tf.expand_dims(
            tf.cast(self._sentence_lengths, tf.float32), -1
        )
        return lstm_outputs_average

    def build_graph(self):
        # embedding layer -> lstm layer -> dense layer -> softmax
        embeddings = self.embedding_layers(self._data)
        lstm_outputs = self.lstm_layer(embeddings)

        weights = tf.get_variable(
            name='final_layer_weigths',
            shape=(self._lstm_size, NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2020)
        )
        biases = tf.get_variable(
            name='final_layer_biases',
            shape=(NUM_CLASSES),
            initializer=tf.random_normal_initializer(seed=2020)
        )

        logits = tf.matmul(lstm_outputs, weights) + biases
        labels_one_hot = tf.one_hot(
            indices=self._labels,
            depth=NUM_CLASSES,
            dtype=tf.float32
        )

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_one_hot,
            logits=logits
        )

        loss = tf.reduce_mean(loss)

        probs = tf.nn.softmax(logits)
        predicted_labels = tf.argmax(probs, axis=1)
        predicted_labels = tf.squeeze(predicted_labels)

        return predicted_labels, loss

    def trainer(self, loss, learning_rate):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return optimizer


def model(train_data_reader, test_data_reader, vocab_size, embedding_size=300,
          lstm_size=50, learning_rate=0.01, num_epochs=5):
    m = train_data_reader._data.shape[0]  # number of training examples
    batch_size = train_data_reader._batch_size  # batch size
    seed = 3
    costs = []  # keep track of the cost

    # create a computational graph
    rnn = RNN(
        vocab_size=vocab_size,
        embedding_size=embedding_size,
        lstm_size=lstm_size,
        batch_size=batch_size
    )
    predicted_labels, loss = rnn.build_graph()
    optimizer = rnn.trainer(loss, learning_rate=learning_rate)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        # Run the initialization
        sess.run(init)
        num_mini_batches = int(m / batch_size)  # number of minibatches of size minibatch_size in the train set

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            seed = seed + 1  # Increases the seed value
            mini_batches = train_data_reader.random_mini_batches(seed)

            for mini_batch in mini_batches:
                # Select a mini batch
                (mini_batch_data, mini_batch_labels, mini_batch_sent_lengths) = mini_batch

                # Run the session to execute the "optimizer" and the "loss"
                _, minibatch_cost = \
                    sess.run([optimizer, loss],
                             feed_dict={rnn._data: mini_batch_data,
                                        rnn._labels: mini_batch_labels,
                                        rnn._sentence_lengths: mini_batch_sent_lengths})

                epoch_cost += minibatch_cost / num_mini_batches

            # print the cost
            print("Cost after epoch {}: {}".format(epoch + 1, epoch_cost))
            costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('epoch')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Divides test data into dev set and test set
        mini_batches_test = test_data_reader.random_mini_batches(0)
        mini_batches_dev = mini_batches_test[:len(mini_batches_test) // 2]
        mini_batches_test = mini_batches_test[len(mini_batches_test) // 2:]

        # accuracy on dev set
        nums_true_pred_dev = 0
        nums_true_pred_test = 0
        for mini_batch_dev in mini_batches_dev:
            (mini_batch_data, mini_batch_labels, mini_batch_sent_lengths) = mini_batch_dev

            dev_plabel_eval = sess.run([predicted_labels],
                                       feed_dict={
                                           rnn._data: mini_batch_data,
                                           rnn._labels: mini_batch_labels,
                                           rnn._sentence_lengths: mini_batch_sent_lengths
                                       })
            matches = np.equal(dev_plabel_eval, mini_batch_labels)
            nums_true_pred_dev += np.sum(matches.astype(float))
        print("Batch size: ", batch_size)
        print("LSTM size: ", lstm_size)
        print("Epoch: ", num_epochs)
        print("Accuracy on dev set: ", nums_true_pred_dev * 100. / (len(mini_batches_dev) * batch_size))

        # accuracy on test set
        for mini_batch_test in mini_batches_test:
            (mini_batch_data, mini_batch_labels, mini_batch_sent_lengths) = mini_batch_test

            test_plabel_eval = sess.run([predicted_labels],
                                        feed_dict={
                                            rnn._data: mini_batch_data,
                                            rnn._labels: mini_batch_labels,
                                            rnn._sentence_lengths: mini_batch_sent_lengths
                                        })
            matches = np.equal(test_plabel_eval, mini_batch_labels)
            nums_true_pred_test += np.sum(matches.astype(float))
        print("Accuracy on test set: ", nums_true_pred_test * 100. / (len(mini_batches_test) * batch_size))
        print('\n')


if __name__ == '__main__':
    # collects and encodes data
    gather_20newsgroups_data()
    encode_data('./datasets/w2v/20news-train-raw.txt', './datasets/w2v/vocab-raw.txt')
    encode_data('./datasets/w2v/20news-test-raw.txt', './datasets/w2v/vocab-raw.txt')

    # vocab size
    with open('./datasets/w2v/vocab-raw.txt') as f:
        vocab_size = len(f.read().splitlines())

    # choose batch_size and lstm_size
    batch_sizes = [32, 64, 128]
    lstm_sizes = [50, 100, 200]
    for batch_size in batch_sizes:
        for lstm_size in lstm_sizes:
            # load data set
            print("Loading train data")
            train_data_reader = DataReader(
                data_path='./datasets/w2v/20news-train-encoded.txt',
                batch_size=batch_size
            )
            print("Loading test data")
            test_data_reader = DataReader(
                data_path='./datasets/w2v/20news-test-encoded.txt',
                batch_size=batch_size
            )
            model(train_data_reader, test_data_reader, vocab_size=vocab_size, lstm_size=lstm_size)

    # best batch_size and lstm_size
    # Batch size: 64
    # LSTM size: 200
    # Epoch: 5
    # Accuracy on dev set: 81.72669491525424
    # Accuracy on test set: 81.48834745762711
