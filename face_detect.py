import cv2 as cv

capture = cv.VideoCapture(0)
face_detect = cv.CascadeClassifier('harr_casscade_classifiers/face.xml')

while(True):
    isTrue, frame=capture.read()

    if isTrue :
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        face_rect = face_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5)
        for (x,y,w,h) in face_rect:
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),thickness=2)
            cv.imshow('detected',frame)
        if cv.waitKey(20) & 0xFF==ord('f'):
            break
    else:
        break


print(len(face_rect))

capture.release()
cv.destroyAllWindows()

