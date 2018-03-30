import re
import string
from os.path import splitext

import itertools
from keras.preprocessing.text import Tokenizer

from commons.utils import save_as_pickle


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
                cleaned_caption = 'startseq ' + cleaned_caption + 'endseq'
                if img_id not in cleaned_captions:
                    cleaned_captions[img_id] = []
                cleaned_captions[img_id].append(cleaned_caption)

        if outfile is not None:
            save_as_pickle(cleaned_captions, outfile)

        return cleaned_captions


class TokenizerTrainer:
    def __init__(self):
        """
        Initialise and train a Tokenizer
        """
        self.tokenizer = Tokenizer(oov_token='UNKNOWN')

    def _remove_rare_words(self):
        """
        Remove words seen less than 5 times
        :return: None
        """
        oov = self.tokenizer.oov_token
        word_counts = list(self.tokenizer.word_counts.items())
        word_counts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in word_counts if wc[1] >= 5]
        indexes = list(range(1, len(sorted_voc) + 1))
        self.tokenizer.word_index = dict(list(zip(sorted_voc, indexes)))
        self.tokenizer.word_index[oov] = len(self.tokenizer.word_index) + 1

    def fit_tokenizer(self, captions, outfile=None):
        """
        Train the tokenizer on the whole captions from Flickr8k
        :return: None
        """
        sentences = list(itertools.chain.from_iterable(list(captions.values())))
        self.tokenizer.fit_on_texts(sentences)
        self._remove_rare_words()

        if outfile is not None:
            save_as_pickle(self.tokenizer, outfile)
