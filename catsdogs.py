import numpy as np
import keras
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

train_path = "C:/Users/Mashrur/Desktop/OpenCV/Faces/train"
test_path = "C:/Users/Mashrur/Desktop/OpenCV/Faces/test"
valid_path = "C:/Users/Mashrur/Desktop/OpenCV/Faces/valid"

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#train 21 -> 40
#test 5 -> 10
#valid -> 16

train_batches = train_datagen.flow_from_directory(train_path, target_size=(128,128), classes=["shouvik","farzad","masud","flora"], batch_size=2)
test_batches = test_datagen.flow_from_directory(test_path, target_size=(128,128), classes=["shouvik","farzad","masud","flora"], batch_size= 1)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(128,128),classes=["shouvik","farzad","masud","flora"], batch_size= 2)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()

model.compile(Adam(lr=0.0001), loss="categorical_crossentropy", metrics = ["accuracy"])
model.fit_generator(train_batches, steps_per_epoch=(22/2), validation_data=valid_batches, validation_steps=(8/2), epochs=80, verbose=2)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Faces/pred/pic1.jpg', target_size = (128, 128))
test_image1 = image.load_img('Faces/pred/pic2.jpg', target_size = (128, 128))
test_image2 = image.load_img('Faces/pred/pic3.jpg', target_size = (128, 128))
test_image3 = image.load_img('Faces/pred/pic4.jpg', target_size = (128, 128))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
r=model.predict(test_image)
q=["shouvik","farzad","masud","flora"]
print(r)
t=r[0][r[0].tolist().index(max(r[0]))]*100
print(q[r[0].tolist().index(max(r[0]))], t, "%")

test_image1 = image.img_to_array(test_image1)
test_image1 = np.expand_dims(test_image1, axis=0)
r=model.predict(test_image1)
q=["shouvik","farzad","masud","flora"]
print(r)
t1=r[0][r[0].tolist().index(max(r[0]))]*100
print(q[r[0].tolist().index(max(r[0]))], t1, "%")

test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis=0)
r=model.predict(test_image2)
q=["shouvik","farzad","masud","flora"]
print(r)
t2=r[0][r[0].tolist().index(max(r[0]))]*100
print(q[r[0].tolist().index(max(r[0]))], t2, "%")

test_image3 = image.img_to_array(test_image3)
test_image3 = np.expand_dims(test_image3, axis=0)
r=model.predict(test_image3)
q=["shouvik","farzad","masud","flora"]
print(r)
t3=r[0][r[0].tolist().index(max(r[0]))]*100
print(q[r[0].tolist().index(max(r[0]))], t3, "%")

print("")
print("AVERAGE:", (t+t1+t2+t3)/4)

#98, 60, 98, 75
#triple dropout 0.1, 0,2, 0.3

from keras.models import load_model
model.save("faces.h5")
