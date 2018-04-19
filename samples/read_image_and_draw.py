# /usr/bin/python
# coding=utf-8
"""functions for reading image frames and drawing
"""

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt
import os.path
import sys

# Change this to where data is stored
data_dir = '../data' 
# vfname = 't24.420.010.left.avi'
vfname = 't24.305.026.left.avi'

if __name__ == '__main__':
    # открыть видеофайл
    cap = cv2.VideoCapture(os.path.join(data_dir, vfname))
    if not cap.isOpened():
        print('error opening video file')
        sys.exit(0)

    # получение кадра
    _, img = cap.read()

    # отображение кадра
    plt.imshow(img)
    plt.show()

    # отображение прямоугольника средствами opencv
    cv2.rectangle(img, (int(img.shape[1] * 0.2), int(img.shape[0] * 0.1)),
                       (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)),
                  (255, 255, 0), 2)
    plt.imshow(img)
    plt.show()