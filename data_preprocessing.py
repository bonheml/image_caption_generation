import argparse
from image_processing import FeatureExtractor
from text_processing import CaptionPreProcessor
from utils import get_image_ids, split_and_save, load_pickle_file

"""
The dataset used for image caption generation is Flickr8K
and the one used for embedding matrix is GloVe

# References
M. Hodosh, P. Young and J. Hockenmaier (2013)
"Framing Image Description as a Ranking Task: Data, Models and 
Evaluation Metrics", Journal of Artificial Intelligence Research,
Volume 47, pages 853-899
http://www.jair.org/papers/paper3994.html

Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. 
GloVe: Global Vectors for Word Representation.
"""


def extract_all(args):
    feature_extractor = FeatureExtractor()
    caption_cleaner = CaptionPreProcessor()
    feature_extractor.extract_all_features(args.images_directory,
                                           args.features_outfile)
    caption_cleaner.preprocess_captions(args.captions_filename,
                                        args.captions_outfile)


def train_test_split(args):
    splits = {'train': get_image_ids(args.train_filename),
              'dev': get_image_ids(args.dev_filename),
              'test': get_image_ids(args.test_filename)}
    captions = load_pickle_file(args.captions_outfile)
    features = load_pickle_file(args.features_outfile)

    for split, ids in splits.items():
        split_and_save(captions, ids,
                       ".".join([args.captions_outfile, split]))
        split_and_save(features, ids,
                       ".".join([args.features_outfile, split]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Preprocess the dataset and save features and cleaned caption
    preprocessor = subparsers.add_parser('preprocess_dataset')
    preprocessor.add_argument('images_directory')
    preprocessor.add_argument('captions_filename')
    preprocessor.add_argument('features_outfile')
    preprocessor.add_argument('captions_outfile')
    preprocessor.set_defaults(func=extract_all)

    # Split the saved features and cleaned captions into train test and dev
    # files
    train_test_split_parser = subparsers.add_parser('train_test_split')
    train_test_split_parser.add_argument('features_outfile')
    train_test_split_parser.add_argument('captions_outfile')
    train_test_split_parser.add_argument('train_filename')
    train_test_split_parser.add_argument('test_filename')
    train_test_split_parser.add_argument('dev_filename')
    train_test_split_parser.set_defaults(func=train_test_split)

    arguments = parser.parse_args()
    arguments.func(arguments)
