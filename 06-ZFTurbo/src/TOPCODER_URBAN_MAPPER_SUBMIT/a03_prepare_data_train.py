# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
import tifffile
import sys


print('Command line arguments: {}'.format(sys.argv))
if len(sys.argv[1:]) == 1:
    INPUT_TRAINING_PATH = sys.argv[1]
else:
    INPUT_TRAINING_PATH = '../input/training/'

OUTPUT_BUILDING_TRAIN = INPUT_TRAINING_PATH + '/train_proc/'
if not os.path.isdir(OUTPUT_BUILDING_TRAIN):
    os.mkdir(OUTPUT_BUILDING_TRAIN)


def convert_tiff_to_png_train():
    files = glob.glob(INPUT_TRAINING_PATH + "/*_RGB.tif")
    print(len(files))
    for f in files:
        id = os.path.basename(f)[:-8]
        print(id)
        dsm_path = os.path.dirname(f) + '/' + id + '_DSM.tif'
        dtm_path = os.path.dirname(f) + '/' + id + '_DTM.tif'
        rgb_path = f
        mask_path = os.path.dirname(f) + '/' + id + '_GTI.tif'
        mask = tifffile.imread(mask_path)
        dsm = tifffile.imread(dsm_path)
        dtm = tifffile.imread(dtm_path)
        rgb = tifffile.imread(rgb_path)

        print('DSM:', dsm.min(), dsm.max())
        print('DTM:', dtm.min(), dtm.max())
        mask[mask > 0] = 255
        mask = mask.astype(np.uint8)

        dsm[dsm > -30000] += 220
        dsm[dsm <= -30000] = 0
        dsm *= 160
        # print('DSM:', dsm.min(), dsm.max())
        dsm = dsm.astype(np.uint16)

        dtm[dtm > -30000] += 100
        dtm[dtm <= -30000] = 0
        dtm *= 540
        # print('DTM:', dtm.min(), dtm.max())
        dtm = dtm.astype(np.uint16)

        print('DSM:', dsm.min(), dsm.max())
        print('DTM:', dtm.min(), dtm.max())

        if mask.shape[:2] != dsm.shape[:2] or dsm.shape[:2] != rgb.shape[:2]:
            print('Shape error!', id, dsm.shape[:2], mask.shape[:2], rgb.shape[:2])

        od = OUTPUT_BUILDING_TRAIN
        mask_path = od + id + '_mask.png'
        dsm_path = od + id + '_dsm.png'
        dtm_path = od + id + '_dtm.png'
        rgb_path = od + id + '_rgb.png'
        cv2.imwrite(mask_path, mask)
        cv2.imwrite(dsm_path, dsm)
        cv2.imwrite(dtm_path, dtm)
        cv2.imwrite(rgb_path, rgb)


def fix_masks():
    files = glob.glob(OUTPUT_BUILDING_TRAIN + '/*_mask.png')
    for f in files:
        print('Go for {}'.format(f))
        mask = cv2.imread(f, 0)
        rgb = cv2.imread(f[:-9] + '_rgb.png')
        fixed_mask = np.zeros(mask.shape, dtype=np.uint8)

        im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        print('Total contours: {}'.format(len(contours)))
        contours_to_fix = []
        for i in range(len(contours)):
            # print(hierarchy[0][i])
            # If parent exists we don't need to draw it
            parent = hierarchy[0][i][3]
            if parent != -1:
                continue

            checker = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(checker, contours, i, (255, 255, 255), cv2.FILLED, cv2.LINE_8, hierarchy)
            # drawContourWithHoles(checker, contours, i, hierarchy)
            nz = np.nonzero(checker)
            count_rgb = 0
            count_mask = len(nz[0])
            for j in range(count_mask):
                sh0 = nz[0][j]
                sh1 = nz[1][j]
                rgb_sum = int(rgb[sh0, sh1, 0]) + int(rgb[sh0, sh1, 1]) + int(rgb[sh0, sh1, 2])
                if rgb_sum > 3:
                    count_rgb += 1

            if count_rgb < 0.9*count_mask:
                # print(count_mask, count_rgb)
                contours_to_fix.append(i)
            else:
                cv2.drawContours(fixed_mask, contours, i, (255, 255, 255), cv2.FILLED, cv2.LINE_8, hierarchy)

        for i in contours_to_fix:
            checker = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(checker, contours, i, (255, 255, 255), cv2.FILLED, cv2.LINE_8, hierarchy)
            # drawContourWithHoles(checker, contours, i, hierarchy)
            nz = np.nonzero(checker)
            count_mask = len(nz[0])
            for j in range(count_mask):
                sh0 = nz[0][j]
                sh1 = nz[1][j]
                rgb_sum = int(rgb[sh0, sh1, 0]) + int(rgb[sh0, sh1, 1]) + int(rgb[sh0, sh1, 2])
                if rgb_sum > 3:
                    fixed_mask[sh0, sh1] = 255

        # diff = np.abs(mask.astype(np.int32) - fixed_mask.astype(np.int32)).astype(np.uint8)
        # show_resized_image(diff)

        # show_resized_image(mask)
        # show_resized_image(fixed_mask)
        print('Fixed contours: {}'.format(len(contours_to_fix)))
        if len(contours_to_fix) > 0:
            cv2.imwrite(f[:-9] + '_mask_fixed.png', fixed_mask)
        else:
            cv2.imwrite(f[:-9] + '_mask_fixed.png', mask)

    print(len(files))


if __name__ == '__main__':
    convert_tiff_to_png_train()
    fix_masks()
