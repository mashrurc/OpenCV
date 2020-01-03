#imports
import numpy as np
from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
print(f"Tensorflow version: {tf.__version__}, Keras version: {keras.__version__}")

list1 = ["T-Shirt", "Shorts", "Jacket", "Long Pants", "Gloves", "Umbrella", "Sweater"]

#% item (T-Shirt, Shorts, T w/Jacket, Long Pants, Gloves, Umbrella, Sweater)
train_l = [
            [0, 0, 100, 100, 80, 3, 50],
            [0, 0, 100, 100, 79, 7, 80],
            [0, 0, 100, 100, 75, 40, 50],
            [0, 0, 100, 100, 70, 90, 75],
            [0, 0, 98, 98, 75, 3, 80],
            [0, 0, 96, 96, 80, 30, 80],
            [0, 0, 95, 95, 80, 40, 80],
            [0, 0, 95, 95, 65, 50, 60],
            [0, 0, 93, 93, 70, 80, 60],
            [3, 2, 80, 95, 50, 20, 80],
            [3, 2, 85, 95, 60, 30, 95],
            [3, 3, 80, 95, 35, 20, 80],
            [4, 4, 70, 95, 20, 15, 93],
            [5, 5, 57, 95, 15, 40, 94],
            [8, 8, 50, 93, 5, 60, 95],
            [8, 10, 45, 90, 0, 0, 90],
            [10, 10, 25, 90, 0, 10, 85],
            [10, 10, 12, 88, 0, 25, 80],
            [12, 12, 5, 87, 0, 90, 76],
            [20, 20, 0, 75, 0, 15, 73],
            [25, 35, 0, 70, 0, 80, 75],
            [40, 40, 0, 50, 0, 55, 65],
            [50, 50, 0, 50, 0, 0, 50],
            [40, 60, 0, 40, 0, 50, 60],
            [70, 70, 0, 30, 0, 0, 20],
            [80, 80, 0, 20, 0, 20, 15],
            [85, 85, 0, 10, 0, 50, 10],
            [90, 90, 0, 10, 0, 70, 20],
            [95, 95, 0, 0, 0, 90, 15]
          ]

#Temp, Wind, Rain %
train_s = [

            [-35,3,3],[-33,5, 10],[-31,0, 30],[-29,9, 80],[-27,11, 5],   [-25,13, 25],[-23,15, 40],[-15,8, 40],[-13,7, 75],[-3,35, 14],[-1,40, 28],[1,12, 18],[3,34, 13], [5,21, 38],[7,16, 50],[9,7, 0],
            [11,3, 5],[13,9, 20],[15,15, 80],[17,27, 12],[19,15, 75],[21,3, 50],[23,9, 0],[25,27, 35],
            [27,5,3],[29,17,18],[31,8,35],[33, 23,60],[35,28,80]

          ]

train_s = np.array(train_s)
train_l = np.array(train_l)

test = [[24,13,42], [-20,5, 80]]
test = np.array(test)

model = Sequential ([
   Dense(4, input_shape=(3,), activation='relu'),
  Dense(8, activation = "relu"),
  Dense(7, activation = "linear")
])

print(model.summary())

model.compile(Adam(lr=0.005), loss="mse", metrics = ["mse"])
model.fit(train_s, train_l, validation_split = 0.05, batch_size=1, epochs=800, verbose = 2)

preds = model.predict(test, batch_size = 1, verbose = 0)
print("")

for i in range(len(preds)):
  for n in range(len(preds[i])):
    if preds[i][n] < 0:
      preds[i][n] = 0
    r=int(preds[i][n].round())
    print(list1[n]+ ":", r, "%")
  print("")

model.save("clothes.h5")

# (20,70), val 0.05, batch 1, e 1000, lr0.005
#0, 0, 96, 96, 80, 30, 80
#20, 20, 0, 75, 0, 15, 73
#90, 90, 0, 10, 0, 70, 20
