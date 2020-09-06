import os

import keras
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras.layers import Input
from keras.models import Model
from keras.utils import multi_gpu_model
from .multi_gpu import CheckPointer

from .metrics import m_iou, m_iou_0, m_iou_1


def train_segmentation_model(segmentation_model,
                             DataLoader,
                             path_to_train_image,
                             path_to_train_labels,
                             path_to_valid_image,
                             path_to_valid_labels,
                             num_gpu=0,
                             workers=30,
                             batch_size=1,
                             learning_rate=3e-4,
                             checkpoint_dir='check_point',
                             weights_path=None,
                             no_epochs=2000,
                             train_input_size=(224, 224),
                             train_input_stride=(224, 224),
                             valid_input_size=(224, 224),
                             valid_input_stride=(224, 224),
                             number_of_class=2,
                             class_weight=None,
                             geohash_precision=None,
                             loss_weights=None,
                             custom_callback=None,
                             custom_loss=None):

    train_generator_params = {
        'x_set_dir': path_to_train_image,
        'y_set_dir': path_to_train_labels,
        'patch_size': train_input_size,
        'patch_stride': train_input_stride,
        'batch_size': batch_size,
        'shuffle': True,
        'is_train': True
    }

    test_generator_params = {
        'x_set_dir': path_to_valid_image,
        'y_set_dir': path_to_valid_labels,
        'patch_size': valid_input_size,
        'patch_stride': valid_input_stride,
        'batch_size': batch_size
    }

    number_of_class = number_of_class
    input_img_shape = (*train_input_size, 3)
    img_input = Input(shape=input_img_shape)
    network_input = [img_input]

    if not (geohash_precision is None):

        train_generator_params['geohash_precision'] = geohash_precision
        test_generator_params['geohash_precision'] = geohash_precision

        input_geohash_shape = (1, 1, geohash_precision)
        geohash_input = Input(shape=input_geohash_shape)
        network_input = [img_input, geohash_input]

    if num_gpu == 1:
        model = segmentation_model(
            network_input, number_of_class=number_of_class)
        model = Model(inputs=network_input, outputs=model)
        print("Training using one GPU..")
    else:
        with tf.device('/cpu:0'):
            model = segmentation_model(
                network_input, number_of_class=number_of_class)
            model = Model(inputs=network_input, outputs=model)
            
    if not (weights_path is None):
        model.load_weights(weights_path, by_name=True)
    if num_gpu > 1:
        parallel_model = multi_gpu_model(model, gpus=num_gpu)
        print("Training using multiple GPUs..")
    else:
        parallel_model = model
        print("Training using one GPU or CPU..")

    # if not (weights_path is None):
    #     parallel_model.load_weights(weights_path, by_name=True)

    if custom_loss is None:
        training_loss = 'categorical_crossentropy'
    else:
        training_loss = custom_loss

    parallel_model.compile(
        loss=training_loss,
        optimizer=keras.optimizers.Adam(learning_rate),
        metrics=["accuracy", m_iou, m_iou_0, m_iou_1],
        loss_weights=loss_weights)

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    tensor_boarder = TensorBoard(log_dir=checkpoint_dir, update_freq='epoch')
    csv_logger = CSVLogger(os.path.join(checkpoint_dir, 'training.log'))
    

    
    checkpointer = CheckPointer(model,checkpoint_dir)
    call_back_list = [tensor_boarder, checkpointer, csv_logger]
    if not (custom_callback is None):
        call_back_list.append(custom_callback)

    train_generator = DataLoader(**train_generator_params)
    test_generator = DataLoader(**test_generator_params)

    parallel_model.fit_generator(
        train_generator,
        epochs=no_epochs,
        workers=workers,
        verbose=1,
        use_multiprocessing=True,
        validation_data=test_generator,
        max_queue_size=workers,
        callbacks=call_back_list,
        class_weight=class_weight)
