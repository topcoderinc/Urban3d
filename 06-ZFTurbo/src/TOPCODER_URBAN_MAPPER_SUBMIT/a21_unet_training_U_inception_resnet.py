# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import os
import sys
import platform

gpu_use = 0
FOLD_TO_CALC = [1, 2, 3, 4, 5]
print('Set gpu to use in tensorflow: {}'.format(gpu_use))
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


print('Command line arguments: {}'.format(sys.argv))
if len(sys.argv[1:]) == 1:
    INPUT_TRAINING_PATH = sys.argv[1]
else:
    INPUT_TRAINING_PATH = '../input/training/'

OUTPUT_PATH = 'modified_data/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
OUTPUT_BUILDING_TRAIN = INPUT_TRAINING_PATH + '/train_proc/'
if not os.path.isdir(OUTPUT_BUILDING_TRAIN):
    os.mkdir(OUTPUT_BUILDING_TRAIN)
MODELS_PATH = 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
HISTORY_FOLDER_PATH = "models/history/"
if not os.path.isdir(HISTORY_FOLDER_PATH):
    os.mkdir(HISTORY_FOLDER_PATH)


import datetime
from a00_common_functions import *
from a00_augmentation_functions import *
from a02_zf_unet_model import *

full_image_list = dict()
full_mask_list = dict()


def get_kfold_split(nfolds):
    from sklearn.model_selection import KFold
    cache_kfold_split = OUTPUT_PATH + 'kfold_cache_{}.pklz'.format(nfolds)
    if not os.path.isfile(cache_kfold_split):
        files = glob.glob(OUTPUT_BUILDING_TRAIN + "/*_mask.png".format(type))
        print('Unique files found: {}'.format(len(files)))
        kf = KFold(n_splits=nfolds, shuffle=True, random_state=66)
        ret = []
        for train_index, test_index in kf.split(range(len(files))):
            ret.append((train_index, test_index))
        save_in_file((files, ret), cache_kfold_split)
    else:
        files, ret = load_from_file(cache_kfold_split)

    return files, ret


def read_initial_data():
    full_image_list = dict()
    full_mask_list = dict()
    files, kfold_split = get_kfold_split(5)
    for f in files:
        name = os.path.basename(f)
        mask_path = f

        # Fixed masks read
        mask_path = f[:-9] + '_mask_fixed.png'

        dsm_path = f[:-9] + '_dsm.png'
        dtm_path = f[:-9] + '_dtm.png'
        rgb_path = f[:-9] + '_rgb.png'
        mask = cv2.imread(mask_path, 0)
        dsm = cv2.imread(dsm_path, 0)
        dtm = cv2.imread(dtm_path, 0)
        rgb = cv2.imread(rgb_path)
        dsm = np.expand_dims(dsm, 2)
        dtm = np.expand_dims(dtm, 2)
        if mask.shape[:2] != dsm.shape[:2] or mask.shape[:2] != rgb.shape[:2]:
            print('Shape error!', name, mask.shape[:2], dsm.shape[:2], rgb.shape[:2])
            exit()
        # show_image(ms)
        # show_image(ph)
        # show_image(mask)
        dsm = dsm.astype(np.float32) / 255.
        dtm = dtm.astype(np.float32) / 255.
        rgb = rgb.astype(np.float32)
        mask = mask.astype(np.float32) / 255.

        img_full = np.concatenate((dsm, dtm, rgb), axis=2)
        full_image_list[name] = img_full.copy()
        full_mask_list[name] = mask.copy()

    return full_image_list, full_mask_list


def batch_generator_train(files, batch_size, augment=True):
    global full_image_list, full_mask_list
    box_size = 288
    positive_num = batch_size // 2

    while True:
        batch_images = []
        batch_masks = []
        batch_files = np.random.choice(files, batch_size)
        total = 0
        for f in batch_files:
            name = os.path.basename(f)
            im_full_big = full_image_list[name]
            im_mask_big = full_mask_list[name]

            if total < positive_num or im_mask_big.sum() < 1:
                # Random box
                start_0 = random.randint(0, im_full_big.shape[0] - box_size)
                start_1 = random.randint(0, im_full_big.shape[1] - box_size)
                end_0 = start_0 + box_size
                end_1 = start_1 + box_size
            else:
                # Get box with at least one positive pixel
                i, j = np.nonzero(im_mask_big)
                ix = np.random.choice(len(i), 1)
                start_0 = max(0, i[ix][0] - random.randint(1, box_size - 1))
                start_1 = max(0, j[ix][0] - random.randint(1, box_size - 1))
                end_0 = start_0 + box_size
                end_1 = start_1 + box_size
                if end_0 > im_mask_big.shape[0]:
                    start_0 = im_mask_big.shape[0] - box_size
                    end_0 = im_mask_big.shape[0]
                if end_1 > im_mask_big.shape[1]:
                    start_1 = im_mask_big.shape[1] - box_size
                    end_1 = im_mask_big.shape[1]
                # print(start_0, start_1, end_0, end_1)

            im_full = im_full_big[start_0:end_0, start_1:end_1]
            im_mask = im_mask_big[start_0:end_0, start_1:end_1]
            if augment:
                im_full, im_mask = random_mirror_with_mask(im_full, im_mask)

            batch_images.append(im_full)
            batch_masks.append(im_mask)
            total += 1

        batch_images = np.array(batch_images, dtype=np.float32)
        # batch_images = np.transpose(batch_images, (0, 3, 1, 2))
        batch_images = preprocess_batch_resnet(batch_images)
        batch_masks = np.array(batch_masks, dtype=np.float32)
        batch_masks = np.expand_dims(batch_masks, axis=1)

        yield batch_images, batch_masks


def train_single_model(num_fold, train_index, test_index, files):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
    from keras.optimizers import Adam, SGD
    from keras import backend as K
    K.set_image_dim_ordering('tf')

    restore = 1
    patience = 50
    epochs = 1000
    optim_type = 'Adam'
    learning_rate = 0.0001
    cnn_type = 'ZF_Seg_Inception_ResNet_v2_288x288'
    print('Creating and compiling {}...'.format(cnn_type))
    model = ZF_Seg_Inception_ResNet_v2_288x288_multi_channel(5)

    final_model_path = MODELS_PATH + '{}_fold_{}.h5'.format(cnn_type, num_fold)
    if os.path.isfile(final_model_path) and restore == 0:
        print('Model already exists for fold {}.'.format(final_model_path))
        return 0.0

    cache_model_path = MODELS_PATH + '{}_temp_fold_{}.h5'.format(cnn_type, num_fold)
    if os.path.isfile(cache_model_path) and restore:
        print('Load model from last point: ', cache_model_path)
        model.load_weights(cache_model_path)
    else:
        print('Start training from begining')

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])
    # model.compile(optimizer=optim, loss='binary_crossentropy', metrics=[dice_coef])

    print('Fitting model...')
    train_files = files[train_index]
    test_files = files[test_index]

    batch_size = 10
    print('Batch size: {}'.format(batch_size))
    steps_per_epoch = 400
    validation_steps = 400
    print('Samples train: {}, Samples valid: {}'.format(steps_per_epoch, validation_steps))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint(cache_model_path, monitor='val_loss', save_best_only=True, verbose=0),
        CSVLogger(HISTORY_FOLDER_PATH + 'history_fold_{}_{}_lr_{}_optim_{}.csv'.format(num_fold,
                                                                                       cnn_type,
                                                                                       learning_rate,
                                                                                       optim_type), append=True)
    ]

    history = model.fit_generator(generator=batch_generator_train(train_files, batch_size),
                                  epochs=epochs,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=batch_generator_train(test_files, batch_size),
                                  validation_steps=validation_steps,
                                  verbose=2,
                                  max_queue_size=30,
                                  callbacks=callbacks)

    min_loss = min(history.history['val_loss'])
    print('Minimum loss for given fold: ', min_loss)
    model.load_weights(cache_model_path)
    model.save(final_model_path)
    now = datetime.datetime.now()
    filename = HISTORY_FOLDER_PATH + 'history_{}_{}_{:.4f}_lr_{}_{}.csv'.format(cnn_type, num_fold, min_loss, learning_rate, now.strftime("%Y-%m-%d-%H-%M"))
    pd.DataFrame(history.history).to_csv(filename, index=False)
    save_history_figure(history, filename[:-4] + '.png')
    return min_loss


def create_models_seg_inception_resnet(nfolds=5):
    files, kfold_split = get_kfold_split(nfolds)
    num_fold = 0
    sum_score = 0
    for train_index, test_index in kfold_split:
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split cars train: ', len(train_index))
        print('Split cars valid: ', len(test_index))

        if 'FOLD_TO_CALC' in globals():
            if num_fold not in FOLD_TO_CALC:
                continue

        score = train_single_model(num_fold, train_index, test_index, np.array(files))
        sum_score += score

    print('Avg loss: {}'.format(sum_score/nfolds))


if __name__ == '__main__':
    num_folds = 5
    full_image_list, full_mask_list = read_initial_data()
    print('Reading data in memory complete!')
    score = create_models_seg_inception_resnet(num_folds)


'''
Model v2:
Fold 1: 0.849972059429
Fold 2: 0.867588068843
Fold 3: 0.86021264568
Fold 4: 0.853568727672
Fold 5: 0.856969092935
'''

