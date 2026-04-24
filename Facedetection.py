import cv2 as cv
from cv2.data import haarcascades

from utils import rescaleFrame
"""
Haar cascade is an algorithim thats already been made to detect images and is trained on images already
"""
#detect faces in images
img = cv.imread('pics/fast.jpg')
resized = rescaleFrame(img,0.25)
gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY) #doesnt look at skin tone only edges
haar_cascade = cv.CascadeClassifier('haar_face.xml')
smile_cascade = cv.CascadeClassifier('haar_smile.xml')
faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5) #returns a list of potential faces

for (x,y,w,h) in faces_rect:
    cv.rectangle(resized,(x,y),(x+w,y+h),(0,255,0),5)
cv.imshow('img',resized)
cv.waitKey(0)
# detect faces in live video
capture = cv.VideoCapture(0)
while True:
    flag, frame = capture.read()
    fram_resized = rescaleFrame(frame, 0.25)

    faces_rect = haar_cascade.detectMultiScale(frame,scaleFactor=1.1,minNeighbors=5)
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
    cv.imshow('img',frame)
    print(f'Number of faces detected: {len(faces_rect)}')
    if cv.waitKey(20) & 0xFF == ord('d'):
        break

capture.release()
cv.destroyAllWindows()
cv.waitKey(0)
