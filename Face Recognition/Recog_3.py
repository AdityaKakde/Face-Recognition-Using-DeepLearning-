import cv2
import numpy as np

recognizer =  cv2.face.LBPHFaceRecognizer_create()
recognizer.read('/home/athreya/recognizer/train.yml')
cascPath = '/home/athreya/PycharmProjects/u1/venv/lib/python3.4/site-packages/cv2/data/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath);
#recognizer.read()


cam = cv2.VideoCapture(0)
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)
c = 0
c1 = 0;
c2 = 0
c3 = 0
c4 = 0
c5 = 0
while True:
    ret, im =cam.read()

    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5 )
    #print("Found {0} faces!".format(len(faces)))



    for(x,y,w,h) in faces:
        c = c + 1
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        print(str(Id)+" "+str(conf))

#From here needs to be optimised by not printing id if conf less than pre-determined tested value
        if(conf<50):
                if(Id==11):
                    Id="Ajay"
                    c1=c1+1
                elif(Id==16):
                    Id="Amu"
                    c2=c2+1
                elif(Id==30):
                    Id="Athreya"
                    c3=c3+1
                elif(Id==8):
                    Id="Aditya"
                    c4=c4+1
        elif(conf>50):
            if (Id == 11):
                Id = "Ajay"
                c1 = c1 + 1
            elif (Id == 16):
                Id = "Amu"
                c2 = c2 + 19
            elif (Id == 30):
                Id = "Athreya"
                c3 = c3 + 1
            elif (Id == 8):
                Id = "Aditya"
                c4 = c4 + 1
        else:
            Id="Unknown"
            c5=c5+1
        cv2.putText(im,str(Id)+","+str(conf), (x,y+h),fontface, fontscale , fontcolor)
    cv2.imshow('im',im)
    if cv2.waitKey(10) == ord('q'):
        print(str(c) + " "+ str(c1)+" "+str(c2)+" "+str(c3)+" "+str(c4)+" "+str(c5))
        print(str(c) + " " + str(c1*100/c) + " " + str(c2*100/c) + " " + str(c3*100/c) + " " + str(c4*100/c) + " " + str(c5))
        f=open("/home/athreya/abc.txt" , "w")
        if(c1>0):
            f.write("Ajay\n")
        if(c2>0):
            f.write("Amu\n")
        if(c3>0):
            f.write("Athreya\n")
        if(c4>0):
            f.write("Aditya\n")
        break
cam.release()
cv2.destroyAllWindows()
