import keras.backend as K
from keras import backend as K
from keras import layers
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Reshape, UpSampling2D,SeparableConv2D)
from keras.models import Model
from keras.regularizers import l2

from .nasnet_keras import NASNetLarge

bn_momentum = 0.9997

def transition_up(x, num_filter, weight_decay=0.0):
    x_1 = Conv2D(
        filters=num_filter,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(weight_decay),
        use_bias=False)(x)
    x_1 = BatchNormalization(axis=-1, epsilon=1e-3, momentum=bn_momentum)(x_1)
    x_1 = Activation('relu')(x_1)


    x_1 = UpSampling2D(size=(2, 2), interpolation='bilinear')(x_1)
    return x_1


def NAS_U_Net(input_tensor, number_of_class, weight_decay=0.0):

    img_input = input_tensor[0]
    base_model = NASNetLarge(
        include_top=False, weights='imagenet', input_tensor=img_input)

    dense_4 = base_model.get_layer('reduction_concat_stem_1').output
    dense_8 = base_model.get_layer('normal_concat_5').output
    dense_16 = base_model.get_layer('normal_concat_12').output
    dense_32 = base_model.get_layer('normal_concat_18').output
    dense_32 = Activation('relu')(dense_32)

    x = transition_up(dense_32,weight_decay=weight_decay, num_filter=256)
    x = Concatenate(axis=-1)([x, dense_16])

    x = transition_up(x,weight_decay=weight_decay, num_filter=128)
    x = Concatenate(axis=-1)([x, dense_8])

    x = transition_up(x,weight_decay=weight_decay, num_filter=64)
    x = Concatenate(axis=-1)([x, dense_4])

    x = Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(weight_decay),
        use_bias=False)(x)
    x = BatchNormalization(axis=-1, epsilon=1e-3, momentum=bn_momentum)(x)
    x = Activation('relu')(x)



    if len(input_tensor)==2:
        geohash_input = input_tensor[1]
        feature_map_height, feature_map_width = K.int_shape(x)[-3:-1]
        geohash_code_feature = UpSampling2D(size=(feature_map_height, feature_map_width),
        interpolation='bilinear')(geohash_input)

        x = Concatenate(axis=-1)([x, geohash_code_feature])

    x = Conv2D(
        filters=number_of_class,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=True,
        padding='same',
        kernel_initializer='he_uniform',
        kernel_regularizer=l2(weight_decay),
        name='last_conv')(x)
    x = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    x = Reshape((-1, number_of_class))(x)
    x = Activation('softmax')(x)

    return x
