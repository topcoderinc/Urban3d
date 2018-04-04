# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import cv2
import random
import pandas as pd


def random_intensity_change(img, max_change):
    img = img.astype(np.float32)
    for j in range(3):
        delta = random.randint(-max_change, max_change)
        img[:, :, j] += delta
    img[img < 0] = 0
    img[img > 255] = 255
    return img


def random_rotate_with_mask(image, mask, max_angle):
    cols = image.shape[1]
    rows = image.shape[0]

    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)
    dst = cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    dst_msk = cv2.warpAffine(mask, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    return dst, dst_msk


def random_mirror_with_mask(image, mask):
    # all possible mirroring and flips
    # (in total there are only 8 possible configurations)
    mirror = random.randint(0, 1)
    if mirror == 1:
        # flipud
        image = image[::-1, :, :].copy()
        mask = mask[::-1, :].copy()
    angle = random.randint(0, 3)
    if angle != 0:
        image = np.rot90(image, k=angle).copy()
        mask = np.rot90(mask, k=angle).copy()
    return image, mask


def get_mirror_image_by_index(image, index):
    if index < 4:
        image = np.rot90(image, k=index)
    else:
        if len(image.shape) == 3:
            image = image[::-1, :, :]
        else:
            image = image[::-1, :]
        image = np.rot90(image, k=index-4)
    return image


def get_mirror_image_by_index_backward(image, index):
    if index < 4:
        image = np.rot90(image, k=-index)
    else:
        image = np.rot90(image, k=-(index-4))
        if len(image.shape) == 3:
            image = image[::-1, :, :]
        else:
            image = image[::-1, :]
    return image

def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def save_history_figure(history, path, columns=('loss', 'val_loss')):
    import matplotlib.pyplot as plt
    s = pd.DataFrame(history.history)
    plt.plot(s[list(columns)])
    plt.savefig(path)
    plt.close()
