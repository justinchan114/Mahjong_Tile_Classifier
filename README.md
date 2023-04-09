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
4. Label 9, 34-41 vary significantly between tiles, it is expected the performance of these tiles could be worse.
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
### Error Analysis
Models were evaluated with f1-score and classification_report. Classes that performed poorly (with f1-score <0.7 or 0.8) were studied and analysed. Observations were made and further data augmentation (RandomZoom) was included. For further details, refer to the jupyter notebook.

## Results
The tuned Mobilenet_V2 model with trained output layer of GlobalAveragePooling2D demonstrated its ability to classify mahjong class, with accuracy ~0.93 on test set. The image dataset has a relatively small size of 700 images and 42 classes, which could lead to insufficient data for training, even with data augmentation. It is noticed that label 9, 34-41 varies significantly between tiles, which may have led to the inaccurate predictions. In fact, all labels with f1-score <0.8 in test set are all from the above mentioned labels.

Data augmentation and base model training has solved the overfitting problem, and significantly improved model accuracy. 
To further improved the accuracy of model, a larger data set, especially on honours (label 34-41) shall be required.
