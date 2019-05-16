import cv2
import numpy as np
cascPath = '/home/athreya/PycharmProjects/u1/venv/lib/python3.4/site-packages/cv2/data/haarcascade_frontalface_default.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

c = cv2.VideoCapture(0)
while(True):
    ret , img = c.read()
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(g,1.3,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img , (x,y) , (x+w,y+h) , (255,0,0) , 1)
    cv2.imshow("Face found",img)
    if(cv2.waitKey(1)==ord('q')):
        break;

