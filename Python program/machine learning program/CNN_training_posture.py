#This neural network was developed with Keras

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,GaussianNoise
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

batch_size = 32
nb_classes = 3
nb_epoch = 50
#data_augmentation = True

# input image dimensions
img_rows, img_cols = 30, 60
# The CIFAR10 images are RGB.
img_channels = 3

surrender_train = np.load("surrender_30x60_train.npy")
weapon_train = np.load("weapon_30x60_train.npy")
neg_train = np.load("neg_posture_train.npy")

surrender_test = np.load("surrender_30x60_test.npy")
weapon_test = np.load("weapon_30x60_test.npy")
neg_test = np.load("neg_posture_test.npy")


X_train = np.concatenate((surrender_train,weapon_train,neg_train),axis=0)
X_test = np.concatenate((surrender_test,weapon_test,neg_test),axis=0)

class_surrender = np.zeros((surrender_train.shape[0],1))
class_surrender.fill(1)
class_weapon = np.zeros((weapon_train.shape[0],1))
class_weapon.fill(2)
class_neg = np.zeros((neg_train.shape[0],1))
class_neg.fill(0)

y_train = np.concatenate((class_surrender,class_weapon,class_neg),axis=0)

class_surrender = np.zeros((surrender_test.shape[0],1))
class_surrender.fill(1)
class_weapon = np.zeros((weapon_test.shape[0],1))
class_weapon.fill(2)
class_neg = np.zeros((neg_test.shape[0],1))
class_neg.fill(0)

y_test = np.concatenate((class_surrender,class_weapon,class_neg),axis=0)

# The data, shuffled and split between train and test sets:
#(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('Y_train shape:', Y_train.shape)
print('Y_test shape:', Y_test.shape)


model = Sequential()

model.add(Convolution2D(32, 3, 3, 
                        input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.75))


# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.75))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
X_train /= 255
X_test /= 255

# datagen = ImageDataGenerator(
#     rotation_range=20,
#     zca_whitening=True,
#     horizontal_flip=True,        
#     # width_shift_range=0.2,
#     # height_shift_range=0.2,
#     # shear_range=0.2,
#     # zoom_range=0.2,
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

model.save('gesture_reg_model4.h5')

loss_and_metrics_train = model.evaluate(X_train, Y_train, batch_size=32)
loss_and_metrics_test = model.evaluate(X_test, Y_test, batch_size=32)

print("Training complete!\n")
print("train: ",loss_and_metrics_train)
print("test: ",loss_and_metrics_test)
