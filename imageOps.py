import cv2
import numpy as np
from random import randint

img = cv2.imread("C:\\Users\\Mashrur\\Desktop\\OpenCV\\bird.jpg", cv2.IMREAD_COLOR)

#px = img[100, 100]
#img[100:200, 100:200] = [randint(0,255), randint(0,255), randint(0,255)]
#v2.rectangle(img, (180, 100), (440, 300), (82, 167, 32), 2)

face = img[100:300, 180:440]
img[0:200, 540: 800] = face
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
