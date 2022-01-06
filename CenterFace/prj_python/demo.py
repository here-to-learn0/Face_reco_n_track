import cv2
import scipy.io as sio
import os
from centerface import CenterFace
import numpy as np

def camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    centerface = CenterFace()
    while True:
        ret, frame = cap.read()
        dets, lms = centerface(frame, h, w, threshold=0.35)
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        for lm in lms:
            for i in range(0, 5):
                cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
        cv2.imshow('out', frame)
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()


def display_image(dets, lms, frame):
    for det in dets:
        boxes, score = det[:4], det[4]
        cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
    
    cv2.imshow('out', frame)
    cv2.waitKey(0)

def get_faces(frame, dets):
    print(frame.shape)
    for det in dets:
        xy = det[:4].astype(int)
        frame = frame[xy[0]: xy[2], xy[1]: xy[3], :]
        # cv2.imshow('out', frame)
        # cv2.waitKey(0)
        print(frame.shape)

    
    
def test_image():
    frame = cv2.imread('test_hd.jpg')
    h, w = frame.shape[:2]
    landmarks = True
    centerface = CenterFace(landmarks=landmarks)
    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
    else:
        dets = centerface(frame, threshold=0.35)
    print(dets.shape)
    # get_faces(frame, dets)


if __name__ == '__main__':
    # np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    camera()
    # test_image()
