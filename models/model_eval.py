import heapq

from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu


class Beam:
    """
    Beam object inspired from this post from M.Tanti:
    https://geekyisawesome.blogspot.ie/2016/10/using-beam-search-to-generate-most.html
    """
    def __init__(self, n_top):
        """
        Initialise Beam object to keep track of n_top sequences
        :param n_top: number of best sequences to keep
        """
        self.n_top = n_top
        self.heap = list()

    def append(self, proba, done, sequence):
        """
        Add a sequence to the heap of Beam object
        :param proba: probability of the sequence
        :param done: True if the sequence is finished, False otherwise
        :param sequence: The sequence
        :return: None
        """
        heapq.heappush(self.heap, (proba, done, sequence))
        if len(self.heap) > self.n_top:
            heapq.heappop(self.heap)

    def __iter__(self):
        """
        Return elements of the heap if iterated on
        :return: heap tuple containing (proba, is_finished, sequence)
        """
        return iter(self.heap)

    def __str__(self):
        heap_str = "Beam search of {} top sequences\n".format(self.n_top)
        for proba, _, seq in self.heap:
            heap_str += "{} with {:.2f} probability\n".format(seq, proba)
        return heap_str


class BeamSearch:
    def __init__(self, model, tokenizer, clip=50, n_top=3,
                 sentence_boundaries=('startseq', 'endseq')):
        """
        Init beam search
        :param model: model used for prediction
        :param tokenizer: tokenizer used for word indexing
        :param clip: maximum caption length
        :param n_top: number of best possible sequences kept each time
        :param sentence_boundaries: tuple of (start, end) of sequence tokens
        """
        self.model = model
        self.tokenizer = tokenizer
        self.clip = clip
        self.n_top = n_top
        self.start_token = sentence_boundaries[0]
        self.end_token = sentence_boundaries[1]
        self.rev_word_idx = {v: k for k, v in tokenizer.word_index.items()}

    def _make_pred(self, features, curr_beam, elem):
        """
        Predict possible next word and append them to beam search.
        Only n_top possibilities will be kept.
        :param features: features of a given image
        :param curr_beam: Beam object currently used
        :param elem: tuple of (probability, done, sequence) from previous beam
        :return: None
        """
        input_text = " ".join(elem[2])
        sequence = self.tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=self.clip)
        y_pred = self.model.predict([features, sequence])[0]
        for i, proba in enumerate(y_pred):
            word = self.rev_word_idx.get(i)
            if word is None:
                continue
            if word == self.end_token:
                curr_beam.append(elem[0] * proba, True, elem[2])
            else:
                curr_beam.append(elem[0] * proba, False, elem[2] + [word])

    def search(self, features):
        """
        Performs beam search from image features
        :param features: features extracted from the image
        :return: (best_caption, probability)
        """
        last_beam = Beam(self.n_top)
        last_beam.append(1.0, False, [self.start_token])
        while True:
            curr_beam = Beam(self.n_top)
            for elem in last_beam:
                if elem[1] is True:
                    curr_beam.append(*elem)
                else:
                    self._make_pred(features, curr_beam, elem)
            (best_proba, done, best_seq) = max(curr_beam)
            print(curr_beam)
            best_seq = best_seq[1:]
            if done is True or len(best_seq) >= self.clip:
                return " ".join(best_seq), best_proba
            last_beam = curr_beam


def get_bleu_score(y_true, y_pred):
    """
    Generate bleu scores for 1 to 4 ngrams
    :param y_true: list of references
    :param y_pred: the caption generated
    :return: None
    """
    bleu_scores = [corpus_bleu(y_true, y_pred, weights=(1.0, 0, 0, 0)),
                   corpus_bleu(y_true, y_pred, weights=(0.5, 0.5, 0, 0)),
                   corpus_bleu(y_true, y_pred, weights=(0.3, 0.3, 0.3, 0)),
                   corpus_bleu(y_true, y_pred, weights=(0.25, 0.25, 0.25, 0.25))
                   ]
    print("Bleu-1\tBleu-2\tBleu-3\tBleu-4")
    print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(*bleu_scores))


def evaluate_model(model, captions, features, tokenizer):
    """
    Evaluate the model by generating a caption for each item in test dataset
    and calculating its bleu scores
    :param model: model to test
    :param captions: dict of test captions
    :param features: dict of extracted features
    :param tokenizer: tokenizer used during model training
    :return:
    """
    y_true = []
    y_pred = []
    start = len('startseq')
    end = -len('endseq')
    beam_search = BeamSearch(model, tokenizer)
    for img_id, caption_list in captions.items():
        y, proba = beam_search.search(features[img_id])
        refs = [c[start:end].split() for c in caption_list]
        y_true.append(refs)
        y_pred.append(y.split())
    get_bleu_score(y_true, y_pred)
