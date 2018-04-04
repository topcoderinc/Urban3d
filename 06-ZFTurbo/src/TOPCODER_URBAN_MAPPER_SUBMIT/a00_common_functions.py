# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import pandas as pd
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
from collections import Counter, defaultdict
from multiprocessing import Process, Manager
import random


random.seed(2016)
np.random.seed(2016)


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def dice(im1, im2, empty_score=1.0):
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


# Somehow it needed close values to standard ResNet preprocessing
def preprocess_batch_resnet(batch):
    batch -= 127
    return batch


def rle(img):
    prev = -1
    str1 = ''
    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img[i, j]
            if i == 0 and j == 0:
                str1 += str(pixel)
                count += 1
                prev = pixel
                continue
            if img[i, j] != prev:
                str1 += ',' + str(count)
                str1 += ',' + str(pixel)
                prev = pixel
                count = 1
            else:
                count += 1
    str1 += ',' + str(count)
    return str1


def img_from_rle(str1, shape):
    arr = str1.split(',')
    mask = np.zeros((shape[0]*shape[1],), dtype=np.uint32)
    current = 0
    for i in range(0, len(arr), 2):
        pixel = int(arr[i])
        counter = int(arr[i+1])
        mask[current:current + counter] = pixel
        current += counter
    mask = np.reshape(mask, (shape[0], shape[1]))
    return mask
