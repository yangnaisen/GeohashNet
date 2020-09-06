import keras.backend as K
from keras import backend as K
from keras import layers
from keras.layers import (Activation, BatchNormalization, Concatenate, Conv2D,
                          Reshape, UpSampling2D, SeparableConv2D)
from keras.models import Model
from keras.regularizers import l2

from .nasnet_keras import NASNetLarge, _normal_a_cell

bn_momentum = 0.9997
interpolation='bilinear'

def transition_up(x, p, num_filter, block_id, weight_decay=0.0):
    x_1, p_1 = _normal_a_cell(x, p, num_filter, block_id=block_id)

    x_1 = UpSampling2D(size=(2, 2), interpolation=interpolation)(x_1)
    return x_1


def NAS_U_Net(input_tensor, number_of_class, weight_decay=0.0):

    img_input = input_tensor[0]
    base_model = NASNetLarge(
        include_top=False, weights='imagenet', input_tensor=img_input)

    normal_cell_4_x = base_model.get_layer('reduction_concat_stem_1').output
    normal_cell_8_x = base_model.get_layer('normal_concat_5').output

    normal_cell_16_x = base_model.get_layer('normal_concat_12').output

    normal_cell_32_x = base_model.get_layer('normal_concat_18').output
    normal_cell_32_p = base_model.get_layer('normal_concat_17').output

    penultimate_filters = 4032
    filters = penultimate_filters // 24
    filter_multiplier = 2

    x = transition_up(
        normal_cell_32_x,
        normal_cell_32_p,
        filters * filter_multiplier,
        '32_to_16',
        weight_decay=weight_decay)

    x = transition_up(
        x, 
        normal_cell_16_x,
        filters, 
        '16_to_8', 
        weight_decay=weight_decay)

    x = transition_up(
        x,
        normal_cell_8_x,
        filters // filter_multiplier,
        '8_to_4',
        weight_decay=weight_decay)

    x, p = _normal_a_cell(
        x,
        normal_cell_4_x,
        filters // (filter_multiplier**2),
        block_id='final_res_4')
    x = Activation('relu')(x)

    if len(input_tensor) == 2:
        geohash_input = input_tensor[1]
        feature_map_height, feature_map_width = K.int_shape(x)[-3:-1]
        geohash_code_feature = UpSampling2D(
            size=(feature_map_height, feature_map_width),
            interpolation=interpolation)(geohash_input)

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
    x = UpSampling2D(size=(4, 4), interpolation=interpolation)(x)
    x = Reshape((-1, number_of_class))(x)
    x = Activation('softmax')(x)

    return x
