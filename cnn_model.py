import cv2 as cv
import numpy
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.regularizers import l2
import os

img_size = 32
X_train = []
Y_train = []
X_test = []
Y_test = []

datadir_training = 'datasets/trainingSet'
datadir_test = 'datasets/test_set'


def convert_to_one_hot(y, c):
    y = numpy.eye(c)[y.reshape(-1)]
    return y


for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    path = os.path.join(datadir_training, i)
    for img in os.listdir(path):
        image = cv.imread(os.path.join(path, img))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV|cv.THRESH_OTSU)[1]
        thresh = cv.resize(thresh, (img_size, img_size))
        X_train.append(thresh)
        Y_train.append(int(i))

for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
    path = os.path.join(datadir_test, i)
    for img in os.listdir(path):
        image = cv.imread(os.path.join(path, img))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
        thresh = cv.resize(thresh, (img_size, img_size))
        X_test.append(thresh)
        Y_test.append(int(i))


X_train = numpy.array(X_train)
X_test = numpy.array(X_test)
Y_train = numpy.array(Y_train)
Y_test = numpy.array(Y_test)

Y_train = convert_to_one_hot(Y_train, 10)
Y_test = convert_to_one_hot(Y_test, 10)

print('Shape of X_train: ' + str(X_train.shape))
print('Shape of Y_train: ' + str(Y_train.shape))
print('Shape of X_test: ' + str(X_test.shape))
print('Shape of Y_test: ' + str(Y_test.shape))


def cnn_model(input_shape=(32, 32, 1), classes=10):
    X_input = Input(input_shape)
    X = Conv2D(16, (5, 5), strides=(1, 1), name='conv1', kernel_initializer=glorot_uniform(seed=0),
               activity_regularizer=l2(0.001))(X_input)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Conv2D(32, (5, 5), strides=(1, 1), name='conv2', kernel_initializer=glorot_uniform(seed=0),
               activity_regularizer=l2(0.001))(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Flatten()(X)
    X = Dense(120, activation='relu', use_bias=False, kernel_initializer=glorot_uniform(seed=0),
              name='fc1', activity_regularizer=l2(0.001))(X)
    X = Dense(84, activation='relu', use_bias=False, kernel_initializer=glorot_uniform(seed=0),
              name='fc2', activity_regularizer=l2(0.001))(X)
    X = Dense(classes, activation='softmax', use_bias=False, kernel_initializer=glorot_uniform(seed=0), name='fc3')(X)

    model = Model(inputs=X_input, outputs=X, name='cnn_model')
    return model


model = cnn_model(input_shape=(32, 32, 1), classes=10)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30, batch_size=1000)
prediction = model.evaluate(X_test, Y_test)
print('loss: ' + str(prediction[0]))
print('accuracy: ' + str(prediction[1]))

model.save('digits_v1.h5')
