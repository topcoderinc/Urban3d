# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import sys
from fixed_nets.inception_resnet_v2 import InceptionResNetV2
sys.setrecursionlimit(5000)


def preprocess_batch(batch):
    batch -= 0.5
    return batch


def dice_coef(y_true, y_pred):
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    from keras import backend as K
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def multi_conv_layer(x, layers, size, dropout, batch_norm):
    from keras.layers import Conv2D
    from keras.layers.normalization import BatchNormalization
    from keras.layers.core import SpatialDropout2D, Activation

    for i in range(layers):
        x = Conv2D(size, (3, 3), padding='same')(x)
        if batch_norm is True:
            x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
    if dropout > 0:
        x = SpatialDropout2D(dropout)(x)
    return x


"""
Unet with Inception Resnet V2 encoder
"""

def ZF_Seg_Inception_ResNet_v2_288x288():
    from keras.models import Model
    from keras.layers.merge import concatenate
    from keras.layers.convolutional import UpSampling2D
    from keras.layers import Conv2D
    from keras import backend as K

    if K.image_dim_ordering() == 'th':
        inputs = (3, 288, 288)
        axis = 1
    else:
        inputs = (288, 288, 3)
        axis = -1

    base_model = InceptionResNetV2(include_top=False, input_shape=inputs, weights='imagenet')

    if 0:
        conv1 = base_model.get_layer('activation_3').output
        conv2 = base_model.get_layer('activation_5').output
        conv3 = base_model.get_layer('block35_10_ac').output
        conv4 = base_model.get_layer('block17_20_ac').output
        conv5 = base_model.get_layer('conv_7b_ac').output
        for i in range(len(base_model.layers)):
            print(i, base_model.layers[i].name)
        exit()
    else:
        conv1 = base_model.layers[9].output
        conv2 = base_model.layers[16].output
        conv3 = base_model.layers[260].output
        conv4 = base_model.layers[594].output
        conv5 = base_model.layers[779].output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=axis)
    conv6 = multi_conv_layer(up6, 2, 256, 0.0, True)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=axis)
    conv7 = multi_conv_layer(up7, 2, 256, 0.0, True)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=axis)
    conv8 = multi_conv_layer(up8, 2, 128, 0.0, True)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=axis)
    conv9 = multi_conv_layer(up9, 2, 64, 0.0, True)

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=axis)
    conv10 = multi_conv_layer(up10, 2, 48, 0.2, True)

    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)

    return model


def ZF_Seg_Inception_ResNet_v2_288x288_multi_channel(input_ch, type='train'):
    from keras.models import Model
    from keras.layers.merge import concatenate
    from keras.layers.convolutional import UpSampling2D
    from keras.layers import Conv2D
    from keras import backend as K

    if K.image_dim_ordering() == 'th':
        inputs_3 = (3, 288, 288)
        inputs_4 = (input_ch, 288, 288)
        axis = 1
    else:
        inputs_3 = (288, 288, 3)
        inputs_4 = (288, 288, input_ch)
        axis = -1

    # Create head for new model
    print('Create head for model...')
    base_model = InceptionResNetV2(include_top=False, input_shape=inputs_4, weights=None)

    if type == 'train':
        print('Create imagenet part for model and copy initial weights...')
        model2 = InceptionResNetV2(include_top=False, input_shape=inputs_3, weights='imagenet')

        # Recalculate weights on first layer
        (weights, ) = model2.layers[1].get_weights()
        new_weights = np.zeros((weights.shape[0], weights.shape[1], input_ch, weights.shape[3]), dtype=np.float32)
        for i in range(input_ch):
            new_weights[:, :, i, :] = weights[:, :, i % 3, :].copy()
        new_weights = new_weights * 3. / input_ch
        base_model.layers[1].set_weights((new_weights.copy(), ))

        # Copy all other weights
        for i in range(2, len(base_model.layers)):
            print('Copy weights {} from {}'.format(i, len(base_model.layers)))
            layer1 = base_model.layers[i]
            layer2 = model2.layers[i]
            layer1.set_weights(layer2.get_weights())

    if 0:
        conv1 = base_model.get_layer('activation_3').output
        conv2 = base_model.get_layer('activation_5').output
        conv3 = base_model.get_layer('block35_10_ac').output
        conv4 = base_model.get_layer('block17_20_ac').output
        conv5 = base_model.get_layer('conv_7b_ac').output
        for i in range(len(base_model.layers)):
            print(i, base_model.layers[i].name)
        exit()
    else:
        conv1 = base_model.layers[9].output
        conv2 = base_model.layers[16].output
        conv3 = base_model.layers[260].output
        conv4 = base_model.layers[594].output
        conv5 = base_model.layers[779].output

    print('Create decoder...')
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=axis)
    conv6 = multi_conv_layer(up6, 2, 256, 0.0, True)

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=axis)
    conv7 = multi_conv_layer(up7, 2, 256, 0.0, True)

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=axis)
    conv8 = multi_conv_layer(up8, 2, 128, 0.0, True)

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=axis)
    conv9 = multi_conv_layer(up9, 2, 64, 0.0, True)

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=axis)
    conv10 = multi_conv_layer(up10, 2, 48, 0.2, True)

    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    print('Finish creating layer structure...')
    model = Model(base_model.input, x)
    print('Model graph construction...')
    return model