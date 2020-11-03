import numpy
import cv2 as cv
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, Dense, Flatten
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.regularizers import l2


def convert_to_one_hot(v, c):
    v = numpy.eye(c)[v.reshape(-1)]
    return v


def load_model(dataset):
    X = []
    Y = []
    for row in open(dataset):
        row = row.split(",")
        label = int(row[0])
        image = numpy.array([int(a) for a in row[1:]], dtype="uint8")
        image = image.reshape((28, 28))
        image = cv.resize(image, (32, 32))
        X.append(image)
        Y.append(label)
    X = numpy.array(X, dtype="float32")
    Y = numpy.array(Y, dtype="int")
    return X, Y


dataset_dir = 'A_Z Handwritten Data.csv'
x, y = load_model(dataset_dir)
x = numpy.expand_dims(x, axis=-1)
y = convert_to_one_hot(y, 26)
print(x.shape)
print(y.shape)


def a_z_model(input_shape=(32, 32, 1), classes=26):
    X_input = Input(input_shape)
    X = Conv2D(32, (3, 3), strides=(1, 1), padding="same", name='conv1',
               kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Conv2D(32, (3, 3), strides=(1, 1), padding="same", name='conv2',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Conv2D(64, (3, 3), strides=(1, 1), padding="same", name='conv3',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Conv2D(128, (3, 3), strides=(1, 1), padding="same", name='conv4',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = Activation(activation='relu')(X)
    X = MaxPooling2D((2, 2), strides=(2, 2))(X)
    X = Flatten()(X)
    X = Dense(120, activation='relu', name='fc1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(60, activation='relu', name='fc2', kernel_initializer=glorot_uniform(seed=0))(X)
    X = Dense(classes, activation='softmax', name='fc3', kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='a_z_model')
    return model


cnn_model = a_z_model(input_shape=(32, 32, 1), classes=26)
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn_model.summary()
cnn_model.fit(x, y, epochs=2, batch_size=10000)
