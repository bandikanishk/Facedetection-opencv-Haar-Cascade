import cv2 as cv

def rescaleFrame(frame, scale=0.75):
    #images, videos, live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dim = (width, height)

    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)
