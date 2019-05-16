import cv2
import os
import numpy as np
from PIL import Image

recognizer =  cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.EigenFacesRecognizer_create()
detector= cv2.CascadeClassifier("/home/athreya/PycharmProjects/u1/venv/lib/python3.4/site-packages/cv2/data/haarcascade_frontalface_default.xml")

path = "/home/athreya/dataSet"

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    Ids=[]
    for imagePath in imagePaths:

        pilImage=Image.open(imagePath).convert('L')

        imageNp=np.array(pilImage,'uint8')


        Id = int(os.path.split(imagePath)[-1].split(".")[1])

        print(Id)
        cv2.imshow("hi", imageNp)
        cv2.waitKey(50)
       # faces=detector.detectMultiScale(imageNp)
        faceSamples.append(imageNp)
        Ids.append(Id)
    return faceSamples,Ids


faceSamples,Ids = getImagesAndLabels(path)
recognizer.train(faceSamples , np.array(Ids))
recognizer.save("/home/athreya/recognizer/train.yml")
cv2.destroyAllWindows()

