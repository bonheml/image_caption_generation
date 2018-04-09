# image caption generation
Generate caption from images

## Setup
This project use Flickr8K dataset (M. Hodosh, P. Young and J. Hockenmaier (2013) "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics", Journal of Artificial Intelligence Research, Volume 47, pages 853-899 http://www.jair.org/papers/paper3994.html).

This dataset is available on demand [here](https://forms.illinois.edu/sec/1713398).
The images and captions of this dataset must be downloaded and unzipped before use.


The dependences should also be installed using `pip install -r requirements.txt` at the root of the project.

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
- `<model_name>` is the name of the model to used for feature extraction. It can be either 'xception', 'VGG16' or 'VGG19'

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
