from keras.applications.nasnet import preprocess_input

from .dataloader_geohash import InriaDataLoaderGeohash


class InriaDataLoaderGeohashNASNet(InriaDataLoaderGeohash):
    def preprocess_input(self, img):
        img = img.astype('float32')
        img = preprocess_input(img)
        return img
