import re
import string
from os.path import splitext

import numpy as np
from keras.preprocessing.text import Tokenizer

from utils import save_as_pickle


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
                cleaned_caption = '<s> ' + cleaned_caption + ' </s>'
                if img_id not in cleaned_captions:
                    cleaned_captions[img_id] = []
                cleaned_captions[img_id].append(cleaned_caption)

        if outfile is not None:
            save_as_pickle(cleaned_captions, outfile)

        return cleaned_captions


class EmbeddingMatrixGenerator:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def generate_embedding_index(self, infile):
        embeddings_index = {}
        texts = []
        with open(infile) as f:
            for line in f:
                texts.append(line)
                values = line.split()
                embeddings_index[values[0]] = np.asarray(values[1:],
                                                         dtype='float32')
        self.tokenizer.fit_on_texts(texts)
        return embeddings_index

    def generate_embedding_matrix(self, embedding_idx, embedding_dim):
        word_idx = self.tokenizer.word_index
        embedding_matrix = np.zeros((len(word_idx) + 1, embedding_dim))
        for word, i in word_idx.items():
            embedding_vector = embedding_idx.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector[:embedding_dim]
        return embedding_matrix

    def generate_embedding(self, embedding_dim, infile, outfile=None):
        embedding_idx = self.generate_embedding_index(infile)
        embedding_matrix = self.generate_embedding_matrix(embedding_idx,
                                                          embedding_dim)
        if outfile is not None:
            save_as_pickle((embedding_matrix, self.tokenizer), outfile)

        return (embedding_matrix, self.tokenizer)
