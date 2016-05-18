import numpy as np
import cv2

# Out cat cascade classifier
cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

# Reads our cat image
cat_img = cv2.imread('cat.jpg')
cat_gray = cv2.cvtColor(cat_img, cv2.COLOR_BGR2GRAY) # Convert to grayscale (since haarcascade uses it)

# Find our cats!
cat_faces = cat_cascade.detectMultiScale(cat_gray)

# Draw rectangle onto region where cat face is found
for (x, y, w, h) in cat_faces:
    cv2.rectangle(cat_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Shows images and destroys window on keypress
cv2.imshow('cats', cat_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
