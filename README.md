# handwritten_digits_and_alphabets_recognition
Handwritten Digits and Alphabets Recognition using Convolutional Neural Networks

The following project is based on recognition of handwritten digits and english characters by using convolutional neural network.
The network was trained on a dataset from MNIST and Char74K  mixed with some personal data accounting upto a 400k samples in dataset.

## Content:

* **A_Z_Handwritten_Data** : dataset of alphabets in csv format
* **digits_dataset_train** : dataset of digit in csv format
* **digits_dataset_test** : dataset for testing in csv format

* **dataset_creation.py**: to make images from a set of images in a single image
* **test.py**: the file predicts the characters on the image saved as "img.jpg"
* **final_model.h5**: The final neural network 
* **A_Z_model.py**: model used for training the neural network for alphabets only
* **final_model.py** : model used for building the neural network for both alphabets and digits
* **cnn_model.py** : model used for building neural network for digits only
* **image_to_csv.py** : for converting dataset of digis (jpg) to csv format 
### Prerequisites :
The following libraries are to be installed on the local machine :
```
->OpenCV
->numpy
->imutils
->keras
->sklearn
```
### Dataset_creation.py:
The follwoing file generates digits dataset from a single jpg image containing the digits ordered from left to right in certain rows.

### Results:
The model trained had an accuracy of 99.50% on training set.

## Creators:
* Yuvnish Malhotra 
* Ishika Budhiraja
* Neelanshu Garg
* Sreeja
