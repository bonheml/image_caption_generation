from keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from numpy import argmax


def generate_description(model, tokenizer, features, max_len=40):
    input_text = 'startseq'
    reverse_word_index = {v:k for k,v in tokenizer.word_index.items()}
    for i in range(max_len):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_len)
        y_pred = model.predict([features, sequence])
        print(y_pred)
        y_pred = argmax(y_pred)
        print(y_pred)
        word = reverse_word_index[y_pred]
        if word is None or word == 'endseq':
            break
        input_text += ' ' + word
    return input_text[9:]


def get_bleu_score(y_true, y_pred):
    bleu_scores = []
    bleu_scores.append(corpus_bleu(y_true, y_pred, weights=(1.0, 0, 0, 0)))
    bleu_scores.append(corpus_bleu(y_true, y_pred, weights=(0.5, 0.5, 0, 0)))
    bleu_scores.append(corpus_bleu(y_true, y_pred, weights=(0.3, 0.3, 0.3, 0)))
    bleu_scores.append(corpus_bleu(y_true, y_pred,
                                   weights=(0.25, 0.25, 0.25, 0.25)))
    bleu_scores = bleu_scores[::-1]
    print("Bleu-1\tBleu-2\tBleu-3\tBleu-4")
    print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(*bleu_scores))


def evaluate_model(model, captions, features, tokenizer, max_len=40):
    y_true = []
    y_pred = []
    for img_id, caption_list in captions.items():
        y = generate_description(model, tokenizer, features[img_id], max_len)
        refs = [c[9:-7].split() for c in caption_list]
        y_true.append(refs)
        y_pred.append(y.split())
    get_bleu_score(y_true, y_pred)