import sys
root_folder = '../../code'
sys.path.append(root_folder)
from net_architecture.nasnet_4x_geohash_SE_mobile import NAS_U_Net #pylint:disable = E0401
from training_tool.train_model import train_segmentation_model #pylint:disable = E0401
from training_tool.loss_function import dice_coef_loss #pylint:disable = E0401
from data_loader.dataloader_geohash_nasnet import InriaDataLoaderGeohashNASNet#pylint:disable = E0401

from utils.keras_config import set_keras_config,get_available_gpus_num #pylint:disable = E0401,E0611
from training_tool.lr_tricks import LearningRateFinder,CyclicCosineRestart #pylint:disable = E0401,E0611

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

split_train_image_folder = os.path.join('../../','dataset','split_random','train','image')
split_train_label_folder = os.path.join('../../','dataset','split_random','train','label')
split_valid_image_folder = os.path.join('../../','dataset','split_random','valid','image')
split_valid_label_folder = os.path.join('../../','dataset','split_random','valid','label')


path_to_train_image = split_train_image_folder
path_to_train_labels = split_train_label_folder                           
path_to_valid_image = split_valid_image_folder
path_to_valid_labels = split_valid_label_folder


weights_path =None
#weights_path = 'check_point_0/epoch_0059_weights.h5py'
#weights_path = 'check_point_1/epoch_0050_weights.h5py'
#weights_path = 'check_point_2/epoch_0195_weights.h5py'
#weights_path = 'check_point_3/epoch_0087_weights.h5py'
#weights_path = 'check_point_1/best_model-0020.hdf5'
#weights_path = 'check_point_2/best_model-0293.hdf5'
#weights_path = 'check_point_3/best_model-0001.hdf5'
#weights_path = 'check_point_4/best_model-0005.hdf5'

if weights_path is None:
    use_warmup=True
else:
    use_warmup=False

lr_callback = CyclicCosineRestart(lr_min=1e-6,lr_max = 1e-4,
                                    number_of_batches=640,number_of_epochs=100,
                                    use_warmup=use_warmup)
#lr_callback = LearningRateFinder(number_of_batches=100)
train_segmentation_model(    NAS_U_Net,
                             InriaDataLoaderGeohashNASNet,
                             path_to_train_image,
                             path_to_train_labels,                             
                             path_to_valid_image,
                             path_to_valid_labels,
                             weights_path = weights_path,
                             num_gpu= 1,
                             workers = 8,
                             batch_size = 16,
                             learning_rate = 1e-4,
                             checkpoint_dir = 'check_point',
                             no_epochs = 405,
                             train_input_size = (512,512),
                             valid_input_size = (512,512),
                             train_input_stride = (512,512),
                            #  train_input_stride = (256,256),
                             valid_input_stride = (512,512),
                             geohash_precision=20,
                             custom_callback =lr_callback,
                             loss_weights = None,
                             custom_loss = None,
                           #   custom_loss = dice_coef_loss,
                             )
