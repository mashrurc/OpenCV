from keras.models import load_model
import numpy as np

new_model = load_model("clothes.h5")

test = np.array([[23,13,0]])

print(new_model.predict(test, batch_size=1, verbose=0))
