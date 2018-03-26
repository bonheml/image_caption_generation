import argparse
from image_processing import FeatureExtractor
from text_processing import CaptionPreProcessor
from utils import get_image_ids, split_and_save, load_features

"""
The dataset used here is Flickr8K
# Reference
M. Hodosh, P. Young and J. Hockenmaier (2013)
"Framing Image Description as a Ranking Task: Data, Models and 
Evaluation Metrics", Journal of Artificial Intelligence Research,
Volume 47, pages 853-899
http://www.jair.org/papers/paper3994.html
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
    captions = load_features(args.captions_outfile)
    features = load_features(args.features_outfile)

    for split, ids in splits.items():
        split_and_save(captions, ids,
                       ".".join([args.captions_outfile, split]))
        split_and_save(features, ids,
                       ".".join([args.features_outfile, split]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pre process dataset')
    parser.add_argument('images_directory')
    parser.add_argument('captions_filename')
    parser.add_argument('train_filename')
    parser.add_argument('test_filename')
    parser.add_argument('dev_filename')
    parser.add_argument('features_outfile')
    parser.add_argument('captions_outfile')
    arguments = parser.parse_args()
    extract_all(arguments)
    train_test_split(arguments)
