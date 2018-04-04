# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import sys
import platform
from skimage import measure

from a00_common_functions import *

PRECISION = 4

if len(sys.argv[1:]) == 1:
    SUBM_FILE = sys.argv[1]
else:
    SUBM_FILE = 'out.txt'


def convert_image_to_text(path):
    img = cv2.imread(path, 0)
    new_mask = np.zeros(img.shape, dtype=np.uint8)
    im2, contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('Total contours: {}'.format(len(contours)))
    small_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        # print(area)
        if area < 100:
            small_area += 1
            continue
        cv2.drawContours(new_mask, [c], 0, (255, 255, 255), -1)

    # show_resized_image(new_mask)
    arr = measure.label(new_mask, connectivity=1)
    str1 = rle(arr)
    # print(str1)
    # tmp_mask = np.zeros(img.shape, dtype=np.uint8)
    # tmp_mask[arr == 1] = 255
    # show_resized_image(tmp_mask)
    # print(arr.min(), arr.max())
    print('Small contours: {}'.format(small_area))
    return str1


def create_submission_from_cache_inception_resnet():
    out_file = SUBM_FILE
    out = open(out_file, 'w')
    cache_path = "../cache_test_inception_resnet/"
    files = glob.glob(cache_path + '*_mask.png')
    print('Masks found: {}'.format(len(files)))
    for f in files:
        name = os.path.basename(f)
        out.write(name[:-17] + '\n')
        mask = cv2.imread(f, 0)
        out.write(str(mask.shape[0]) + ',' + str(mask.shape[1]) + '\n')
        str1 = convert_image_to_text(f)
        out.write(str1 + '\n')
    out.close()


if __name__ == '__main__':
    create_submission_from_cache_inception_resnet()


# LB: 879781
