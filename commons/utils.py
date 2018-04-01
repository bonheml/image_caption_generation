from os.path import splitext
from pickle import load, dump


def load_pickle_file(filename):
    """
    Load pickle file containing features
    :param filename: Name of the pickle file to load
    :return: loaded features
    """
    with open(filename, mode='rb') as f:
        features = load(f)
    return features


def save_as_pickle(to_save, outfile):
    """
    Save data as pickle file
    :param to_save: data to save
    :param outfile: filename of the pickle file
    :return: None
    """
    with open(outfile, 'wb') as out:
            dump(to_save, out)


def get_image_ids(filename):
    """
    Get image ids from a file and remove the jpg extension
    :param filename: filename of the file containing the image ids
    :return: dict of image ids
    """
    with open(filename) as f:
        image_ids = [splitext(i)[0] for i in f.read().split('\n') if i]
    return image_ids


def split_and_save(features, ids, outfile):
    """
    Filter features with id in ids and save them in file outfile
    :param features: whole features dictionary
    :param ids: keys to keep
    :param outfile: filename to save the filtered dict
    :return: None
    """
    features = {image_id: features[image_id] for image_id in ids}
    save_as_pickle(features, outfile)
