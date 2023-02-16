# Mahjong_Classifier
Image classification of Mahjong tiles with Tensor, with transfer learning from MobilnetV2.

## Data set
The data set consist of 700 centered Mahjong images, in 42 classes. Majority of them are from https://github.com/Camerash/mahjong-dataset, with minor amendments and addition of ~80 images. The label and label ID of the images are stored in "data.csv"

## Methodology
### Exploratory Data Analysis
The data quality, structure and raw images were first checked. The data.csv was imported into a pandas dataframe to check if there is any abnormality. Few observations were made and corrected upon checking:
1. Incorrect label & label ID pair were rectified
2. Image with different image ratio is cropped, to allow uniform image reshape in later stage
3. Misspelt labels were corrected
### Train test split
The data set is splited in to train test set (80/20) for further processing. Since each label has limited number of samples, stratified split is used to ensure all labels are included in train/test set.
### Model Training
The train set is used for model training. Stratified K Fold cross validation was used, and five models were trained and compared.

## Model Training and Error Analysis
The major steps included in the model training are followings:-
1. Training of output layer and the layer before, with base model(MobilnetV2) frozen (trainable = False)
2. Training of base model
3. Data augmentation to mitigate overfitting
4. Error analysis on the models
This has been an iterative process, and multiple trails were made before errors were resolved.

