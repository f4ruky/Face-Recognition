import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    for(x,y,w,h) in face:
        cv2.rectangle(img, (x,y),(x+w, y+h),(0,255,0),4)
        roi_gray= gray[y:y+h, x:x+h]
        roi_color= img[y:y+h, x:x+h]



    cv2.imshow("Video",img)
    if cv2.waitKey(30) & 0xFF == ord("q"):

     break



cap.release()
cv2.destroyAllWindows()
