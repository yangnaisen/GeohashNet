from .data_loader import InriaDataLoader
from skimage.io import imread
import numpy as np


class InriaDataLoaderInMemory(InriaDataLoader):
    def __init__(self,
                 x_set_dir,
                 y_set_dir,
                 patch_size,
                 patch_stride,
                 batch_size,
                 shuffle=False,
                 is_train=False,
                 num_classes=2):

        super(InriaDataLoaderInMemory, self).__init__(
            x_set_dir,
            y_set_dir,
            patch_size,
            patch_stride,
            batch_size,
            shuffle=shuffle,
            is_train=is_train,
            num_classes=2)
        self.images_files = []
        for filename in self.images_filenames:
            img = imread(filename, plugin='gdal').astype(np.uint8)
            self.images_files.append(img)

        self.labels_files = []
        for filename in self.labels_filenames:
            img = imread(filename, plugin='gdal').astype(np.uint8)
            self.labels_files.append(img)

    def get_patch(self, filenames, patch_idx):
        img_idx = int(patch_idx / self.patches_per_img)
        img_patch_idx = patch_idx % self.patches_per_img
        row_idx = int(img_patch_idx / self.patch_cols_per_img)
        col_idx = img_patch_idx % self.patch_cols_per_img

        if filenames[img_idx].split('/')[-2] == 'image':
            img = self.images_files[img_idx]
        else:
            img = self.labels_files[img_idx]

        if len(img.shape) > 2:
            patch_image = img[row_idx * self.patch_height_stride:row_idx *
                              self.patch_height_stride +
                              self.patch_height, col_idx *
                              self.patch_width_stride:col_idx *
                              self.patch_width_stride +
                              self.patch_width, :].copy()
        else:
            patch_image = img[row_idx * self.patch_height_stride:row_idx *
                              self.patch_height_stride +
                              self.patch_height, col_idx *
                              self.patch_width_stride:col_idx *
                              self.patch_width_stride +
                              self.patch_width].copy()
        return patch_image
