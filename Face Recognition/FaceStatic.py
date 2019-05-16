import cv2
import sys

imagePath = "/home/athreya/Downloads/abba.png"
cascPath = '/home/athreya/PycharmProjects/u1/venv/lib/python3.4/site-packages/cv2/data/haarcascade_frontalface_default.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(7,7)
    )

print("Found {0} faces!".format(len(faces)))  #No of faces


for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#cv2.imshow("Faces found", image)
cv2.waitKey(0)
