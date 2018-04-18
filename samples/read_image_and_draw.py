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

if __name__ == '__main__':
    # открыть видеофайл
    cap = cv2.VideoCapture(os.path.join(data_dir, 't24.305.025.left.avi'))
    if not cap.isOpened():
        print('error opening video file')
        sys.exit(0)

    # получение кадра
    _, img = cap.read()

    # отображение кадра
    plt.imshow(img)
    plt.show()

    # отображение прямоугольника средствами opencv
    cv2.rectangle(img, (img.shape[1] * 0.2, img.shape[0] * 0.1),
                       (img.shape[1] * 0.5, img.shape[0] * 0.5),
                  (255, 255, 0), 2)
    plt.imshow(img)
    plt.show()