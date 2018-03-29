from keras import Input, Model
from keras.layers import Dropout, Dense, Embedding, LSTM, add

from commons.utils import load_pickle_file


def build_merge_model(embedding, features_shape):
    """
    Build merge model from Marc Tanti, et al. in their 2017 papers

    # References:
    Where to put the Image in an Image Caption Generator, 2017.
    What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption
    Generator?, 2017.

    :param embedding: pickle file containing embedding matrix and tokenizer
    :param features_shape: shape of the features extracted from the images
    :return: Merge model
    """
    embedding_matrix, tokenizer = load_pickle_file(embedding)
    vocab_size = len(tokenizer.word_index)

    # Features block
    features_input = Input(shape=features_shape, name="features_input")
    features_dropout = Dropout(0.5, name='features_dropout')(features_input)
    features_dense = Dense(256, activation='relu',
                           name='features_dense')(features_dropout)

    # Sequence block
    sequence_input = Input(shape=(40,), name='sequence_input')
    sequence_embedding = Embedding(vocab_size, embedding_matrix.shape[1],
                                   weights=[embedding_matrix], mask_zero=True,
                                   name='sequence_embedding')(sequence_input)
    sequence_dropout = Dropout(0.5, name='sequence_dropout')(sequence_embedding)
    sequence_lstm = LSTM(256, name='sequence_lstm')(sequence_dropout)

    # Merge block
    merge_add = add([features_dense, sequence_lstm], name='merge_add')
    merge_dense = Dense(256, activation='relu', name='merge_dense')(merge_add)
    merge_output = Dense(vocab_size, activation='softmax',
                         name='merge_output')(merge_dense)

    # Wrap it into a model taking [features, sequence] as parameter and having
    # word index as output
    model = Model(inputs=[features_input, sequence_input],
                  outputs=merge_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
