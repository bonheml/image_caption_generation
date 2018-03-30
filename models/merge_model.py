from keras import Input, Model
from keras.layers import (Dropout, Dense, Embedding, GRU, BatchNormalization,
                          concatenate)

from commons.utils import load_pickle_file


def build_merge_model(tokenizer, features_shape):
    """
    Model based on merge model from Marc Tanti, et al. in their 2017 papers

    # References:
    Where to put the Image in an Image Caption Generator, 2017.
    What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption
    Generator?, 2017.

    :param tokenizer: pickle file containing the tokenizer
    :param features_shape: shape of the features extracted from the images
    :return: Merge model
    """
    tokenizer = load_pickle_file(tokenizer)
    vocab_size = len(tokenizer.word_index) + 1
    layer_size = 128

    # Image block
    image_input = Input(shape=features_shape, name='image_input')
    image_norm = BatchNormalization(input_shape=features_shape,
                                    name='image_normalisation')(image_input)
    image_output = Dense(layer_size, name='image_output')(image_norm)

    # Text generation block
    lang_input = Input(shape=(50,), name='sequence_input')
    lang_embedding = Embedding(vocab_size, layer_size, mask_zero=True,
                               input_length=50,
                               name='sequence_embedding')(lang_input)
    lang_gru = GRU(layer_size, name='sequence_gru')(lang_embedding)
    lang_output = Dropout(0.5, name='sequence_output')(lang_gru)

    # Merge block
    wrapper_merge = concatenate([image_output, lang_output],
                                name='concatenation_layer')
    wrapper_output = Dense(vocab_size, activation='softmax',
                           name='word_prediction')(wrapper_merge)

    model = Model(inputs=[image_input, lang_input], outputs=wrapper_output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model
