import argparse

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.utils import plot_model

from commons.utils import load_pickle_file
from models.Sequencer import Sequencer
from models.merge_model import build_merge_model
from models.model_eval import evaluate_model


def train(args):
    epochs = 1000
    batch_size = 100
    train_samples = 306404
    test_samples = 50903
    train_generator = Sequencer(args.train_captions, args.train_features,
                                args.tokenizer, batch_size)
    test_generator = Sequencer(args.test_captions, args.test_features,
                               args.tokenizer, batch_size)
    model = build_merge_model(args.tokenizer, (2048,))
    model.summary()
    plot_model(model, to_file='model.png', show_shapes=True)
    model_file = 'model.epoch_{epoch:02d}-loss_{val_loss:.2f}.hdf5'
    if args.models_directory:
        filepath = args.models_directory + '/' + model_file
    else:
        filepath = model_file
    checkpoint = ModelCheckpoint(filepath, save_best_only=True)
    model.fit_generator(train_generator.generate_sequences(),
                        steps_per_epoch=train_samples//batch_size,
                        epochs = epochs,
                        validation_data=test_generator.generate_sequences(),
                        validation_steps=test_samples//batch_size,
                        callbacks=[checkpoint, EarlyStopping(patience=4)])


def evaluate(args):
    captions = load_pickle_file(args.captions)
    features = load_pickle_file(args.features)
    tokenizer = load_pickle_file(args.tokenizer)[1]
    model = load_model(args.model)
    evaluate_model(model, captions, features, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Train the model
    trainer = subparsers.add_parser('train')
    trainer.add_argument('tokenizer')
    trainer.add_argument('train_captions')
    trainer.add_argument('test_captions')
    trainer.add_argument('train_features')
    trainer.add_argument('test_features')
    trainer.add_argument('-d', '--models_directory')
    trainer.set_defaults(func=train)

    # Evaluate the model
    evaluator = subparsers.add_parser('evaluate')
    evaluator.add_argument('model')
    evaluator.add_argument('captions')
    evaluator.add_argument('features')
    evaluator.add_argument('tokenizer')
    evaluator.set_defaults(func=evaluate)

    arguments = parser.parse_args()
    arguments.func(arguments)
