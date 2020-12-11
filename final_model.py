import numpy
import cv2 as cv
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dense, Flatten
from keras.initializers import glorot_uniform
from keras.models import Model
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
import csv


def convert_to_one_hot(v, c):
    v = numpy.eye(c)[v.reshape(-1)]
    return v


X = []
Y = []

with open('A_Z Handwritten Data.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 0:
            label = int(row[0])
            image = numpy.array([int(a) for a in row[1:]], dtype="uint8")
            image = image.reshape((28, 28))
            image = cv.resize(image, (32, 32))
            X.append(image)
            Y.append(label)

print("length of X: ", len(X))
print("length of Y: ", len(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = X_train[:100000]
Y_train = Y_train[:100000]
X_test = X_test[:25000]
Y_test = Y_test[:25000]

print("length of X_train: ", len(X_train))
print("length of Y_train: ", len(Y_train))
print("length of X_test: ", len(X_test))
print("length of Y_test: ", len(Y_test))

x_train_digits = []
y_train_digits = []
with open('digits_dataset_train.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 0:
            label = int(row[0])
            image = numpy.array([int(a) for a in row[1:]], dtype="uint8")
            image = image.reshape((32, 32))
            x_train_digits.append(image)
            y_train_digits.append(label)

print("length of x_train_digits: ", len(x_train_digits))
print("length of y_train_digits: ", len(y_train_digits))
x_train_digits, _, y_train_digits, _ = train_test_split(x_train_digits, y_train_digits, train_size=34999)
print("length of x_train_digits: ", len(x_train_digits))
print("length of y_train_digits: ", len(y_train_digits))

for a in x_train_digits:
    X_train.append(a)
for b in y_train_digits:
    Y_train.append(b)
print("length of X_train: ", len(X_train))
print("length of Y_train: ", len(Y_train))

x_test_digits = []
y_test_digits = []
with open('digits_dataset_test.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 0:
            label = int(row[0])
            image = numpy.array([int(a) for a in row[1:]], dtype="uint8")
            image = image.reshape((32, 32))
            x_test_digits.append(image)
            y_test_digits.append(label)

print("length of x_test_digits: ", len(x_test_digits))
print("length of y_test_digits: ", len(y_test_digits))
x_test_digits, _, y_test_digits, _ = train_test_split(x_test_digits, y_test_digits, train_size=8725)
print("length of x_test_digits: ", len(x_test_digits))
print("length of y_test_digits: ", len(y_test_digits))

for k in x_test_digits:
    X_test.append(k)
for m in y_test_digits:
    Y_test.append(m)
print("length of X_test: ", len(X_test))
print("length of Y_test: ", len(Y_test))

X_train = numpy.array(X_train)
Y_train = numpy.array(Y_train)
X_test = numpy.array(X_test)
Y_test = numpy.array(Y_test)

X_train = numpy.expand_dims(X_train, axis=-1)
Y_train = convert_to_one_hot(Y_train, 36)

X_test = numpy.expand_dims(X_test, axis=-1)
Y_test = convert_to_one_hot(Y_test, 36)

print("Shape of X_train: ", X_train.shape)
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)


def final_model(input_shape=(32, 32, 1), classes=36):
    X_input = Input(input_shape)
    Z = Conv2D(16, (5, 5), strides=(1, 1), padding='same', name='conv1',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(0.0001))(X_input)
    Z = Activation(activation='relu')(Z)
    Z = MaxPooling2D((2, 2), strides=(2, 2))(Z)
    Z = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv2',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(0.0001))(Z)
    Z = Activation(activation='relu')(Z)
    Z = MaxPooling2D((2, 2), strides=(2, 2))(Z)
    Z = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv3',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(0.0001))(Z)
    Z = Activation(activation='relu')(Z)
    Z = MaxPooling2D((2, 2), strides=(2, 2))(Z)
    Z = Conv2D(64, (5, 5), strides=(1, 1), padding='same', name='conv4',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(0.0001))(Z)
    Z = Activation(activation='relu')(Z)
    Z = MaxPooling2D((2, 2), strides=(2, 2))(Z)
    Z = Flatten()(Z)
    Z = Dense(120, activation='relu', name='fc1',
              kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(0.0001))(Z)
    Z = Dense(84, activation='relu', name='fc2',
              kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(0.0001))(Z)
    Z = Dense(classes, activation='softmax', name='fc3',
              kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=l2(0.0001))(Z)

    complete_model = Model(inputs=X_input, outputs=Z, name='final_model')
    return complete_model


combine_model = final_model(input_shape=(32, 32, 1), classes=36)
combine_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
combine_model.summary()
combine_model.fit(X_train, Y_train, epochs=50, batch_size=1000)
predictions = combine_model.evaluate(X_test, Y_test)
print("Loss: ", predictions[0])
print("Accuracy: ", predictions[1])
combine_model.save('final_model.h5')
