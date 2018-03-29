from os import listdir
from os.path import abspath, splitext

from keras import Model
from keras.applications import Xception, VGG16, MobileNet, VGG19
from keras.applications.xception import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from commons.utils import save_as_pickle


class FeatureExtractor:
    def __init__(self, model_name='xception'):
        self.model = self._prepare_model(model_name)

    def _prepare_model(self, model_name):
        """
        Initialise the feature extractor
        Here we remove the prediction layer in order to perform the feature
        extraction only.
        """
        models = {"xception": Xception, "VGG16": VGG16, "VGG19": VGG19,
                  "mobilenet": MobileNet}
        if model_name not in models:
            raise NotImplementedError
        model = models[model_name]()
        model.layers.pop()
        model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
        return model

    def extract_features(self, input_file):
        """
        Load image from file 'input_file' with size 299x299
        convert it to an array and reshape it to use it with Xception model
        compute the intern pre processing needed by Xception and predict the
        features of the image
        :param input_file:
        :return:
        """
        filename = abspath(input_file)
        img = load_img(filename, target_size=(299, 299))
        vectorized_img = img_to_array(img)
        vectorized_img = vectorized_img.reshape((1, vectorized_img.shape[0],
                                                 vectorized_img.shape[1],
                                                 vectorized_img.shape[2]))
        preprocessed_img = preprocess_input(vectorized_img)
        features = self.model.predict(preprocessed_img, verbose=0)
        return features

    def extract_all_features(self, directory, outfile=None):
        """
        Extract features from all images in the folder 'directory'
        :param directory: image folder
        :param outfile: Optional filename used to save the features.
        If None no save will be performed.
        :return: dict of features
        """
        features = {}
        i = 0

        for file in listdir(directory):
            if i > 2:
                break
            img_id = splitext(file)[0]
            file_path = "/".join([directory, file])
            features[img_id] = self.extract_features(file_path)
            i += 1

        if outfile is not None:
            save_as_pickle(features, outfile)

        return features
