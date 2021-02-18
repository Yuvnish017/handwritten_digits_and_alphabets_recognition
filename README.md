# handwritten_digits_and_alphabets_recognition
Handwritten Digits and Alphabets Recognition using Convolutional Neural Networks

The following project is based on recognition of handwritten digits and english characters by using convolutional neural network.
The network was trained on a dataset from MNIST(for digits 0-9) and subset of NIST dataset combined with some other sources for 
alphabets(A-Z) (dataset can be found on https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format)  
mixed with some personal data accounting upto a 400k samples in dataset.

## Content:

* **A_Z_Handwritten_Data.csv** : dataset of alphabets in csv format
* **digits_dataset_train.csv** : dataset of digit in csv format
* **digits_dataset_test.csv** : dataset for testing in csv format

* **dataset_creation.py**: to make images from a set of images in a single image
* **test.py**: the file predicts the characters on the image saved as "img.jpg"
* **cnn_model.py** : model used for building neural network for digits only
* **A_Z_model.py**: model used for training the neural network for alphabets only
* **final_model.py** : model used for building the neural network for both alphabets and digits
* **image_to_csv.py** : for converting dataset of digis (jpg) to csv format
* **models** : contains python codes for two different neural network architecture for training along with trained models saved as .h5 files.
			   Names of saved model file in .h5 format describe the architecture of neural network along with accuracy obtained with that model. 
			   For example, conv_34_64_64_256_fc_120_84_36_acc_99.h5 shows that architecture has 4 convolutional layers with 32, 64, 64, 256 filters respectively, 
			   and has 3 fully connected layers with 120, 84 and 36 hidden units. acc_99 in the end specifies that training accuracy obtained with this model is 99%. 
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
The training accuracy for 4 convolutional layers network and 5 convolutional layers network is 99% for both
but 5 convolutional layers network perform better at testing time.

## Creators:
* Yuvnish Malhotra 
* Ishika Budhiraja
* Neelanshu Garg
* Sreeja
