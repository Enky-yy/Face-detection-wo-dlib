import cv2 as cv

eyes_detect = cv.CascadeClassifier('harr_casscade_classifiers/eye.xml')
capture = cv.VideoCapture(0)

while True:
    isTrue ,frame = capture.read()

    if isTrue:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        eye_rect = eyes_detect.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7)

        for (x,y,w,h) in eye_rect:
            cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            cv.imshow('detected',frame)
        if cv.waitKey(20) & 0xff==ord('f'):
            break
    else:
        break

print(len(eye_rect))

capture.release()
cv.destroyAllWindows()
