import cv2
import numpy as np

img = cv2.imread("C:\\Users\\Mashrur\\Desktop\\OpenCV\\bird.jpg", cv2.IMREAD_COLOR)

#          image, start,           end,          color,   width
#cv2.line(img, (200, 200), (600, 600), (82, 167, 32), 5)
#cv2.rectangle(img, (100,100), (700, 700), (82, 167, 32), 5)
#cv2.circle(img, (350,320),  250, (82, 167, 32), 5)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
