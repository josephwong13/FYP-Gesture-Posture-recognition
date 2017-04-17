#This neural network was developed with Keras

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,GaussianNoise
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

batch_size = 68
nb_epoch = 100

nb_classes = 1
# input image dimensions
img_rows, img_cols = 30, 60
img_channels = 3

pos_train = np.load("pos_train.npy")
neg_train = np.load("neg_train.npy")
pos_test = np.load("pos_test.npy")
neg_test = np.load("neg_test.npy")

X_train = np.concatenate((pos_train,neg_train),axis=0)
X_test = np.concatenate((pos_test,neg_test),axis=0)
Y_train = np.concatenate((np.ones((pos_train.shape[0],1)),np.zeros((neg_train.shape[0],1))),axis=0)
Y_test = np.concatenate((np.ones((pos_test.shape[0],1)),np.zeros((neg_test.shape[0],1))),axis=0)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)

model = Sequential()

model.add(Convolution2D(32, 3, 3, 
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
X_train /= 255
X_test /= 255

# datagen = ImageDataGenerator(
#     rotation_range=40,
#     zca_whitening=True,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     zoom_range=0.2,
#     shear_range=0.2,
#     fill_mode='nearest'
#     )

# datagen.fit(X_train)

# model.fit_generator(datagen.flow(X_train,Y_train,
#                                 batch_size=batch_size),
#                                 samples_per_epoch=X_train.shape[0],
#                                 nb_epoch=nb_epoch,
#                                 validation_data=(X_test, Y_test)
#                                 )

model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test),
              shuffle=True)

model.save('human_reg_modeltest.h5')

loss_and_metrics_train = model.evaluate(X_train, Y_train, batch_size=32)
loss_and_metrics_test = model.evaluate(X_test, Y_test, batch_size=32)

print("Training complete!\n")
print("train: ",loss_and_metrics_train)
print("test: ",loss_and_metrics_test)
