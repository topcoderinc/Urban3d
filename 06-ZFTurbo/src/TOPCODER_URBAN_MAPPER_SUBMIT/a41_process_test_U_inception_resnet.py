# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import sys
import platform

gpu_use = 0
print('Set gpu to use in tensorflow: {}'.format(gpu_use))
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

from a00_common_functions import *
from a02_zf_unet_model import *
from a00_augmentation_functions import *
# from a31_validation_U_inception_resnet import get_mask_from_model_v1


PRECISION = 6

CACHE_PATH_TEST = "../cache_test_inception_resnet/"
if not os.path.isdir(CACHE_PATH_TEST):
    os.mkdir(CACHE_PATH_TEST)

print('Command line arguments: {}'.format(sys.argv))
if len(sys.argv[1:]) == 2:
    INPUT_TRAINING_PATH = sys.argv[1]
    INPUT_TESTING_PATH = sys.argv[2]
else:
    INPUT_TRAINING_PATH = '../input/training/'
    INPUT_TESTING_PATH = '../input/testing/'

OUTPUT_PATH = 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
OUTPUT_BUILDING_TEST = INPUT_TESTING_PATH + '/test_proc/'
if not os.path.isdir(OUTPUT_BUILDING_TEST):
    os.mkdir(OUTPUT_BUILDING_TEST)
MODELS_PATH = 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)


def read_initial_data_test():
    global INPUT_PATH
    full_image_list = dict()
    glob_path = OUTPUT_BUILDING_TEST + "/*_rgb.png"
    print('Glob path: {}'.format(glob_path))
    files = glob.glob(glob_path)
    for f in files:
        name = os.path.basename(f)
        dsm_path = f[:-8] + '_dsm.png'
        dtm_path = f[:-8] + '_dtm.png'
        rgb_path = f[:-8] + '_rgb.png'
        dsm = cv2.imread(dsm_path, 0)
        dtm = cv2.imread(dtm_path, 0)
        rgb = cv2.imread(rgb_path)
        dsm = np.expand_dims(dsm, 2)
        dtm = np.expand_dims(dtm, 2)
        dsm = dsm.astype(np.float32) / 255.
        dtm = dtm.astype(np.float32) / 255.
        rgb = rgb.astype(np.float32)

        img_full = np.concatenate((dsm, dtm, rgb), axis=2)
        full_image_list[name] = img_full.copy()

    return full_image_list


def get_mask_from_model_v1(model, image):
    # STEP = 54
    # BORDER = 50
    STEP = 144
    BORDER = 40
    USE_AUGM = 5

    box_size = 288
    final_mask = np.zeros(image.shape[:2], dtype=np.float32)
    count = np.zeros(image.shape[:2], dtype=np.float32)
    image_list = []
    params = []

    # 288 cases
    size_of_subimg = 288
    step = STEP
    for j in range(0, image.shape[0], step):
        for k in range(0, image.shape[1], step):
            start_0 = j
            start_1 = k
            end_0 = start_0 + size_of_subimg
            end_1 = start_1 + size_of_subimg
            if end_0 > image.shape[0]:
                start_0 = image.shape[0] - size_of_subimg
                end_0 = image.shape[0]
            if end_1 > image.shape[1]:
                start_1 = image.shape[1] - size_of_subimg
                end_1 = image.shape[1]

            image_part = image[start_0:end_0, start_1:end_1].copy()
            for i in np.random.choice(list(range(8)), USE_AUGM, replace=False):
                im = get_mirror_image_by_index(image_part.copy(), i)
                # im = cv2.resize(im, (box_size, box_size), cv2.INTER_LANCZOS4)
                image_list.append(im)
                params.append((start_0, start_1, size_of_subimg, i))

    image_list = np.array(image_list, dtype=np.float32)
    image_list = preprocess_batch_resnet(image_list)
    # image_list = np.transpose(image_list, (0, 3, 1, 2))
    mask_list = model.predict(image_list, batch_size=16)
    print('Number of masks:', mask_list.shape)

    border = BORDER
    for i in range(mask_list.shape[0]):
        # Tensorflow version:
        mask = mask_list[i, :, :, 0].copy()
        if mask_list[i].shape[0] != params[i][2]:
            mask = cv2.resize(mask, (params[i][2], params[i][2]), cv2.INTER_LANCZOS4)
        mask = get_mirror_image_by_index_backward(mask, params[i][3])
        # show_resized_image(255*mask)

        # Find location of mask. Cut only central part for non border part
        if params[i][0] < border:
            start_0 = params[i][0]
            mask_start_0 = 0
        else:
            start_0 = params[i][0] + border
            mask_start_0 = border

        if params[i][0] + params[i][2] >= final_mask.shape[0] - border:
            end_0 = params[i][0] + params[i][2]
            mask_end_0 = mask.shape[0]
        else:
            end_0 = params[i][0] + params[i][2] - border
            mask_end_0 = mask.shape[0] - border

        if params[i][1] < border:
            start_1 = params[i][1]
            mask_start_1 = 0
        else:
            start_1 = params[i][1] + border
            mask_start_1 = border

        if params[i][1] + params[i][2] >= final_mask.shape[1] - border:
            end_1 = params[i][1] + params[i][2]
            mask_end_1 = mask.shape[1]
        else:
            end_1 = params[i][1] + params[i][2] - border
            mask_end_1 = mask.shape[1] - border

        try:
            final_mask[start_0:end_0, start_1:end_1] += mask[mask_start_0:mask_end_0, mask_start_1:mask_end_1]
        except:
            print(start_0, end_0, start_1, end_1)
            print(mask_start_0, mask_end_0, mask_start_1, mask_end_1)
            exit()
        count[start_0:end_0, start_1:end_1] += 1

    if count.min() == 0:
        print('Some uncovered parts of image!')

    final_mask /= count
    # show_resized_image(255 * final_mask)
    return final_mask


def process_tst_resnet(nfolds, reversed1=False):

    restore_from_cache = 1
    full_image_list = read_initial_data_test()
    proc_files = sorted(full_image_list.keys())
    if reversed1 == True:
        proc_files = proc_files[::-1]

    print('Files to process: {}'.format(len(proc_files)))
    thr = 0.5
    cnn_type = 'ZF_Seg_Inception_ResNet_v2_288x288'
    for i in range(nfolds):
        print('Read model {} fold {}'.format(cnn_type, i+1))
        start_time = time.time()
        model = ZF_Seg_Inception_ResNet_v2_288x288_multi_channel(5, 'test')
        print('Reading complete in {} seconds'.format(time.time() - start_time))

        start_time = time.time()
        final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, i+1)
        print('Load weights: {}'.format(final_model_path))
        model.load_weights(final_model_path)
        print('Loading weights complete in {} seconds'.format(time.time() - start_time))

        for name in proc_files:
            print('Go for {}...'.format(name))
            cache_path = CACHE_PATH_TEST + name + '_fold_{}.pklz'.format(i)
            if not os.path.isfile(cache_path) or restore_from_cache == 0:
                start_time = time.time()
                tf = full_image_list[name]
                msk = get_mask_from_model_v1(model, tf.copy())
                msk = np.round(msk, PRECISION)
                save_in_file(msk, cache_path)
                print('Complete in {} seconds'.format(time.time() - start_time))
            else:
                print('Restore from cache...')

    for name in proc_files:
        msk = []
        for i in range(nfolds):
            cache_path = CACHE_PATH_TEST + name + '_fold_{}.pklz'.format(i)
            msk_part = load_from_file(cache_path)
            msk.append(msk_part)

        # We store all folds separately
        msk = np.array(msk).mean(axis=0)

        cv2.imwrite(CACHE_PATH_TEST + name + '_prob.png', (255 * msk).astype(np.uint8))
        pred_msk = msk.copy()
        pred_msk[pred_msk > thr] = 1
        pred_msk[pred_msk <= thr] = 0
        if 0:
            show_resized_image((255 * msk))
            show_resized_image((255 * pred_msk))
        cv2.imwrite(CACHE_PATH_TEST + name + '_mask.png', (255 * pred_msk).astype(np.uint8))


if __name__ == '__main__':
    print('Working directories')
    print('OUTPUT_BUILDING_TEST: {}'.format(OUTPUT_BUILDING_TEST))
    print('MODELS_PATH: {}'.format(MODELS_PATH))
    process_tst_resnet(5, False)

