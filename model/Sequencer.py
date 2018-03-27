import itertools
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from commons.utils import load_pickle_file


class Sequencer:
    def __init__(self, captions, features, tokenizer, batch_size):
        """
        Load the cleaned captions, features, tokenizer needed for sequence
        generation and set the batch size of the generator as batch_size
        :param captions: file containing cleaned captions dictionary
        :param features: file containing the dictionary of features
        :param tokenizer: file containing the tokenizer
        :param batch_size: size of each batch yield by generate_sequences
        """
        self.captions = load_pickle_file(captions)
        self.features = load_pickle_file(features)
        self.tokenizer = load_pickle_file(tokenizer)[1]
        self.max_len = self._get_max_caption_length()
        self.vocab_len = len(self.tokenizer.word_index) + 1
        self.batch_size = batch_size

    def _get_max_caption_length(self):
        """
        Get the longest caption size from all captions
        :return: maximum caption size
        """
        captions = list(itertools.chain.from_iterable(self.captions.values()))
        return max([len(c) for c in captions])

    def generate_sequences(self):
        """
        Infinite generator which yield batches of sequences
        of size self.batch_size.
        Each element of a batch is formatted as follows:
        ([X1, X2], y) where X1 is the input sequence, X2 the image features and
        y the expected sequence.
        :return: generator
        """
        res = [[], [], []]
        while True:
            for img_id, descriptions in self.captions.items():
                sequences = self.tokenizer.texts_to_sequences(descriptions)
                for seq in sequences:
                    seq_size = len(seq)
                    for i in range(1, seq_size):
                        if len(res[0]) == self.batch_size:
                            yield ([np.array(res[0]), np.array(res[1])],
                                   np.array(res[2]))
                            res = [[], [], []]
                        padded_seq = pad_sequences([seq[:i]],
                                                   maxlen=self.max_len)[0]
                        output_seq = to_categorical([seq[i]],
                                                    num_classes=self.vocab_len)[0]
                        res[0].append(padded_seq)
                        res[2].append(to_categorical(output_seq))
                        res[1].append(self.features[img_id])