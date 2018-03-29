import re
import string
from os.path import splitext

import numpy as np
from keras.preprocessing.text import Tokenizer

from commons.utils import save_as_pickle
from nltk.corpus import brown


class CaptionPreProcessor:
    def __init__(self):
        """
        Init the Pre processing of captions
        """
        self._punct_regex = re.compile('[%s]' % re.escape(string.punctuation))
        self._digit_trans = str.maketrans('', '', string.digits)

    def clean_caption(self, caption):
        """
        Transform the caption to lowercase, remove punctuation and numbers.
        :param caption: Caption to clean
        :return: Cleaned caption
        """
        caption = " ".join([t for t in caption.split() if len(t) > 1])
        caption = caption.lower()
        caption = self._punct_regex.sub("", caption)
        caption = caption.translate(self._digit_trans)
        return caption

    def preprocess_captions(self, infile, outfile=None):
        """
        Clean all captions from the file 'infile'
        :param infile: The file where the captions are
        :param outfile: Optional filename used to save the features.
        If None no save will be performed.
        :return: dict of cleaned captions
        """
        cleaned_captions = {}

        with open(infile) as f:
            for line in f:
                tokens = line.split()
                if len(tokens) < 2:
                    continue
                img_id = splitext(tokens[0])[0]
                caption = " ".join(tokens[1:])
                cleaned_caption = self.clean_caption(caption)
                cleaned_caption = 'startseq ' + cleaned_caption + ' endseq'
                if img_id not in cleaned_captions:
                    cleaned_captions[img_id] = []
                cleaned_captions[img_id].append(cleaned_caption)

        if outfile is not None:
            save_as_pickle(cleaned_captions, outfile)

        return cleaned_captions


class EmbeddingMatrixGenerator:
    def __init__(self, max_vocab):
        """
        Initialise and train a Tokenizer retaining max_vocab words
        :param max_vocab: vocabulary size of the tokenizer
        """
        self.tokenizer = Tokenizer()
        self.train_tokenizer()
        self._correct_word_index(max_vocab)

    def _correct_word_index(self, max_vocab):
        """
        This function is a workaround to get only max_vocab word indexed
        as tokenizer num_word parameter is not behaving as expected
        see https://github.com/keras-team/keras/issues/8092 for more details
        :param max_vocab: max vocabulary to keep
        :return: None
        """
        self.tokenizer.word_index = {e: i for e, i in
                                     self.tokenizer.word_index.items() if
                                     i <= max_vocab}
        self.tokenizer.word_index[self.tokenizer.oov_token] = max_vocab + 1

    def train_tokenizer(self):
        """
        Train the tokenizer on brown corpus
        Beware, the brown corpus should be downloaded using the command
        nltk.download('brown') beforehand
        :return: None
        """
        sentences = [" ".join(s) for s in brown.sents()]
        self.tokenizer.fit_on_texts(sentences)

    def generate_embedding_index(self, infile):
        """
        Generate indexes for embedding matrix using GloVe file
        :param infile: Glove file
        :return: dictionary of words and their GloVe vectors
        """
        embeddings_index = {}
        with open(infile) as f:
            for line in f:
                values = line.split()
                embeddings_index[values[0]] = np.asarray(values[1:],
                                                         dtype='float32')
        return embeddings_index

    def generate_embedding_matrix(self, embedding_idx, embedding_dim):
        """
        Generate an embedding matrix
        :param embedding_idx: dictionary word/vector pairs
        :param embedding_dim: size of the vector used for each word
        representation
        :return: embedding matrix
        """
        word_idx = self.tokenizer.word_index
        embedding_matrix = np.zeros((len(word_idx), embedding_dim))
        for word, i in word_idx.items():
            embedding_vector = embedding_idx.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector[:embedding_dim]
        return embedding_matrix

    def generate_embedding(self, embedding_dim, infile, outfile=None):
        """
        Generate an embedding matrix and if outfile is not None save it along
        with the tokenizer used for word indexing to avoid changes in tokenizer
        word indexes and retraining.
        :param embedding_dim:
        :param infile: File used to read the Glove vectors
        :param outfile: pickle target file. If None, embedding and tokenizer
        will simply be returned
        :return: (embedding, tokenizer used)
        """
        embedding_idx = self.generate_embedding_index(infile)
        embedding_matrix = self.generate_embedding_matrix(embedding_idx,
                                                          embedding_dim)
        if outfile is not None:
            save_as_pickle((embedding_matrix, self.tokenizer), outfile)

        return embedding_matrix, self.tokenizer
