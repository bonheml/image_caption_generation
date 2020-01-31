# image caption generation
Generate caption from images based on the architecture proposed by Tanti, M., Gatt, A. and Camilleri, K., 2017, September. What is the Role of Recurrent Neural Networks (RNNs) in an Image Caption Generator?. In Proceedings of the 10th International Conference on Natural Language Generation (pp. 51-60).


## Setup
This project use Flickr8K dataset (M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artificial Intelligence Research, Volume 47, pages 853-899 http://www.jair.org/papers/paper3994.html).

This dataset is available on demand [here](https://forms.illinois.edu/sec/1713398).
The images and captions of this dataset must be downloaded and unzipped before use.


The dependencies should also be installed using `pip install -r requirements.txt` at the root of the project.

## Data pre processing

The data pre processing is separated in four steps:
- Clean captions
- Extract image features
- Split the dataset into train, test and dev samples
- Train a tokenizer for word vectorization

The two first steps can be performed using 
`python data_preprocessor.py preprocess_dataset <images_folder> <captions_filename> <features_outfile> <captions_outfile> -m <model_name>`
where: 
- `<image_folder>` is the path to the directory containing Flickr8k images (usually named Flicker8k_Dataset) 
- `<captions_filename>` is the path to the file containing Flickr8K captions (usually named Flickr8k.token.txt)
- `<features_outfile>` is the path of the file used to save the extracted features
- `<captions_outfile>` is the path of the file used to save the cleaned captions
- `<model_name>` is the name of the model to use for feature extraction. It can be either 'xception' (the default value), 'VGG16' or 'VGG19'

The third step can be performed using
`python data_preprocessor.py train_test_split <features_outfile> <captions_outfile> <train_filename> <test_filename> <dev_filename>`
where:
- `<features_outfile>` is path of the file previously used to save the extracted features
- `<captions_outfile>` is path of the file previously used to save the cleaned captions
- `<train_filename>` is the path of the file containing all the image ID of the train dataset (usually named Flickr_8k.trainImages.txt)
- `<test_filename>` is the path of the file containing all the image ID of the test dataset (usually named Flickr_8k.testImages.txt)
- `<dev_filename>` is the path of the file containing all the image ID of the dev dataset (usually named Flickr_8k.devImages.txt)

The last step can be performed using
`python data_preprocessor.py fit_tokenizer <caption_file> <output_file>`
where:
- `<caption_file>` is the path of the file previously used to save the cleaned captions
- `<output_file>` is the path of the file used to save the trained tokenizer


## Model training and evaluation

Once the data preprocessing steps performed, the model can be trained and evaluated.


The training is done using `python model_manager.py train <tokenizer> <train_captions> <test_captions> <train_features> <test_features> -m <model_name>`
where:
- `<tokenizer>` is the path of the file used to save the tokenizer
- `<train_captions>` is the path of the file used to save the captions of the training sample (the file name is the one of the original caption file with a .train extension)
- `<dev_captions>` is the path of the file used to save the captions of the dev sample (the file name is the one of the original caption file with a .dev extension).
- `<train_features>` is the path of the file used to save the features of the training sample (the file name is the one of the original features file with a .train extension)
- `<dev_features>` is the path of the file used to save the features of the dev sample (the file name is the one of the original features file with a .dev extension)
- `<model_name>` is the name of the model used for feature extraction. It can be either 'xception' (the default value), 'VGG16' or 'VGG19'

During the training the best results will be saved along with charts of the loss history, the accuracy history and the model architecture.


The evaluation is performed on a trained model using `python model_manager.py evaluate <model> <captions> <features> <tokenizer>`
where:
- `<model>` is the name of the model to evaluate
- `<captions>` is the path of the file containing the captions to use as references (the test set should be used here)
- `<features>` is the path of the file containing the features to use (the test set should be used here)
- `<tokenizer>` is the path of file containing the trained tokenizer to use
