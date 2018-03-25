from os.path import splitext
from pickle import load, dump


def load_features(filename, images_ids):
    """
    Load pickle file containing features
    :param filename: Name of the pickle file to load
    :param images_ids: List of image ids
    :return: dict of features for each image id.
    """
    with open(filename, mode='rb') as f:
        features = load(f)
    return {image_id: features[image_id] for image_id in images_ids}


def save_features(features, outfile):
    """
    Save features as pickle file
    :param features: features to save
    :param outfile: filename of the pickle file
    :return: None
    """
    with open(outfile, 'wb') as out:
            dump(features, out)


def get_image_ids(filename):
    """
    Get image ids from a file and remove the jpg extension
    :param filename: filename of the file containing the image ids
    :return: dict of image ids
    """
    with open(filename) as f:
        image_ids = [splitext(i)[0] for i in f.read().split('\n') if i]
    return image_ids


def split_and_save(infile, ids, outfile):
    """
    Get element from file infile with id in ids and save them in file outfile
    :param infile: file containing the data to split and save
    :param ids: key to keep
    :param outfile: filename to save the filtered dict
    :return: None
    """
    features = load_features(infile, ids)
    save_features(features, outfile)