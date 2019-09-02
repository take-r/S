import os
import sys
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import pandas as pd

class DataLoader():
    def __init__(self,
                 data_dir,
                 data_split,
                 preprocessing,
                 **kwargs):

        self.data_dir = data_dir
        self.data_split = data_split
        self.preprocessing = preprocessing

        self.split_data()

        self.X_train = self.load_data(self.X_train_files)
        self.y_train = self.load_data(self.y_train_files)
        self.X_valid = self.load_data(self.X_valid_files)
        self.y_valid = self.load_data(self.y_valid_files)
        self.X_test = self.load_data(self.X_test_files)
        self.y_test = self.load_data(self.y_test_files)

        self.y_train = np.expand_dims(self.y_train[:,:,:,0], axis=-1)
        self.y_valid = np.expand_dims(self.y_valid[:,:,:,0], axis=-1)
        self.y_test = np.expand_dims(self.y_test[:,:,:,0], axis=-1)

        assert(self.X_train.shape[0] == self.y_train.shape[0])
        assert(self.X_valid.shape[0] == self.y_valid.shape[0])
        assert(self.X_test.shape[0] == self.y_test.shape[0])

        print('# of training data: {}'.format(self.X_train.shape[0]))
        print('# of validation data: {}'.format(self.X_valid.shape[0]))
        print('# of test data: {}'.format(self.X_test.shape[0]))

        if self.preprocessing['flag']:
            self.preprocess()

        if False:
            self.X_train = self.X_train[:10]
            self.y_train = self.y_train[:10]
            self.X_valid = self.X_valid[:10]
            self.y_valid = self.y_valid[:10]
            self.X_test = self.X_test[:10]
            self.y_test = self.y_test[:10]


    def split_data(self):
        df = pd.read_csv(self.data_split)

        self.train_idxes = list(df[df['split']=='train']['idx'].values)
        self.valid_idxes = list(df[df['split']=='val']['idx'].values)
        self.test_idxes = list(df[df['split']=='test']['idx'].values)

        self.X_train_files = [os.path.join(self.data_dir, 'image', 'image_{:04d}.png'.format(idx)) for idx in self.train_idxes]
        self.y_train_files = [os.path.join(self.data_dir, 'label', 'label_{:04d}.png'.format(idx)) for idx in self.train_idxes]
        self.X_valid_files = [os.path.join(self.data_dir, 'image', 'image_{:04d}.png'.format(idx)) for idx in self.valid_idxes]
        self.y_valid_files = [os.path.join(self.data_dir, 'label', 'label_{:04d}.png'.format(idx)) for idx in self.valid_idxes]
        self.X_test_files = [os.path.join(self.data_dir, 'image', 'image_{:04d}.png'.format(idx)) for idx in self.test_idxes]
        self.y_test_files = [os.path.join(self.data_dir, 'label', 'label_{:04d}.png'.format(idx)) for idx in self.test_idxes]

    def load_data(self, files):

        images = []
        for filepath in tqdm(files):
            image = cv2.imread(filepath)
            images.append(image[:, :, ::-1])

        return np.array(images)

    def preprocess(self):

        #TODO: change for RGB images
        def normalize(x, mean=None, std=None):
            if mean is None:
                mean = np.mean(x)
            if std is None:
                std = np.std(x)
            x = (x-mean)/std
            return x, mean, std

        if self.preprocessing['normalization']:
            self.X_train, self.mean, self.std = normalize(self.X_train, mean=None, std=None)
            self.X_valid, _, _ = normalize(self.X_valid, mean=self.mean, std=self.std)
            self.X_test, _, _ = normalize(self.X_test, mean=self.mean, std=self.std)

            with open(self.data_dir+'/train_mean_std.txt', 'w') as f:
                f.write('mean:{}\nstd:{}'.format(self.mean, self.std))

        else:
            pass


if __name__=='__main__':
    # sanity check

    import matplotlib.pyplot as plt

    config = {
        'data_dir':'../camvid_data',
        'data_split':'../data_split_table.csv',
        'preprocessing': {
                'flag': True,
                'normalization': False,
                'min_max': False
                }
            }
    dataloader = DataLoader(**config)

    plt.imshow(dataloader.X_train[0])
    plt.show()
