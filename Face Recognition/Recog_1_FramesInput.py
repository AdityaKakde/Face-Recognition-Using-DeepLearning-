import cv2
import os
import numpy as np
from PIL import Image
cascPath = '/home/athreya/PycharmProjects/u1/venv/lib/python3.4/site-packages/cv2/data/haarcascade_frontalface_alt.xml'

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

id = input('Enter the usn')
s=0
c = cv2.VideoCapture(0)
while(True):
    ret , img = c.read()
    g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(g,1.1,5)

    for(x,y,w,h) in faces:
        s=s+1
        cv2.imwrite("/home/athreya/dataSet/User." + str(id) + "." + str(s) + ".jpg", g[y:y+h,x:x+w])
        cv2.waitKey(150)
        cv2.rectangle(img , (x,y) , (x+w,y+h) , (255,0,0) , 1)
    cv2.imshow("Face found",img)

    if(cv2.waitKey(1)==ord('q')):
        break
    if s>75:
        break



c.release()
cv2.destroyAllWindows()
