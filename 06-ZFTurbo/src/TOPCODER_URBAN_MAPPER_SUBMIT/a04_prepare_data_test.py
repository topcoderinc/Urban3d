# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

from a00_common_functions import *
import tifffile
import sys

print('Command line arguments: {}'.format(sys.argv))
if len(sys.argv[1:]) == 1:
    INPUT_TESTING_PATH = sys.argv[1]
else:
    INPUT_TESTING_PATH = '../input/testing/'

OUTPUT_BUILDING_TEST = INPUT_TESTING_PATH + '/test_proc/'
if not os.path.isdir(OUTPUT_BUILDING_TEST):
    os.mkdir(OUTPUT_BUILDING_TEST)


def convert_tiff_to_png_tst():
    files = glob.glob(INPUT_TESTING_PATH + "/*_RGB.tif")
    print(len(files))
    for f in files:
        id = os.path.basename(f)[:-8]
        print(id)
        dsm_path = os.path.dirname(f) + '/' + id + '_DSM.tif'
        dtm_path = os.path.dirname(f) + '/' + id + '_DTM.tif'
        rgb_path = f
        dsm = tifffile.imread(dsm_path)
        dtm = tifffile.imread(dtm_path)
        rgb = tifffile.imread(rgb_path)

        print('DSM:', dsm.min(), dsm.max())
        print('DTM:', dtm.min(), dtm.max())

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

        if dsm.shape[:2] != rgb.shape[:2]:
            print('Shape error!', id, dsm.shape[:2], rgb.shape[:2])

        od = OUTPUT_BUILDING_TEST
        dsm_path = od + id + '_dsm.png'
        dtm_path = od + id + '_dtm.png'
        rgb_path = od + id + '_rgb.png'
        cv2.imwrite(dsm_path, dsm)
        cv2.imwrite(dtm_path, dtm)
        cv2.imwrite(rgb_path, rgb)


if __name__ == '__main__':
    print('Working directories')
    print('INPUT_TESTING_PATH: {}'.format(INPUT_TESTING_PATH))
    print('OUTPUT_BUILDING_TEST: {}'.format(OUTPUT_BUILDING_TEST))
    convert_tiff_to_png_tst()
