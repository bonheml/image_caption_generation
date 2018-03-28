import argparse

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import plot_model

from models.Sequencer import Sequencer
from models.merge_model import build_merge_model


def train_model(args):
    epochs = 1000
    batch_size = 100
    steps = epochs/batch_size
    train_generator = Sequencer(args.train_captions, args.train_features,
                                args.tokenizer, batch_size)
    test_generator = Sequencer(args.test_captions, args.test_features,
                                args.tokenizer, batch_size)
    model = build_merge_model(args.tokenizer, (2048,))
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    model_file = ('model.epoch_{epoch:02d}-loss_{val_loss:.2f}.hdf5')
    if args.models_directory:
        filepath = args.models_directory + '/' + model_file
    else:
        filepath = model_file
    checkpoint = ModelCheckpoint(filepath, save_best_only=True)
    model.fit_generator(train_generator.generate_sequences(),
                        steps_per_epoch=steps, epochs = epochs,
                        validation_data=test_generator.generate_sequences(),
                        validation_steps=steps,
                        callbacks=[checkpoint, EarlyStopping(patience=4)])


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
    trainer.set_defaults(func=train_model)

    arguments = parser.parse_args()
    arguments.func(arguments)
