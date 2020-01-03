from keras.models import load_model
model = load_model('faces.h5')

import numpy as np
from keras.preprocessing import image

import os, os.path
DIR = 'C:/Users/Mashrur/Desktop/OpenCV/unknownfaces'

def run(index):
    test_image = image.load_img( 'C://Users/Mashrur/Desktop/OpenCV/unknownfaces/face.jpg', target_size = (128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    r=model.predict(test_image)

    print(r)
    q=["shouvik","farzad","masud","flora"]
    t=r[0][r[0].tolist().index(max(r[0]))]*100
    print(q[r[0].tolist().index(max(r[0]))], t, "%")

index=0
run(index)
