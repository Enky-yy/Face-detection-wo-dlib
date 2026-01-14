import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0)
face_detect = cv.CascadeClassifier('harr_casscade_classifiers/face.xml')
eyes_detect = cv.CascadeClassifier('harr_casscade_classifiers/eye.xml')

while(True):
    isTrue, frame=capture.read()

    if isTrue :
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        face_rect = face_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        eyes_rect = eyes_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        for (x,y,w,h) in face_rect:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
            cv.imshow('detected',frame)
        for (x,y,w,h) in eyes_rect:
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv.imshow('detected',frame)
        # comb_img = np.concatenate((x,y),axis=1)
        # cv.imshow('detected',comb_img)
        if cv.waitKey(20) & 0xFF==ord('f'):
            break
    else:
        break


print(len(face_rect))
print(len(eyes_rect))

capture.release()
cv.destroyAllWindows()

