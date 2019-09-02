import os
import sys
import warnings
import numpy as np
from copy import copy
import math

from model.vanilla_unet import UNet
from image_augmentor import ImageAugmentor
from results_logger import Logger

from keras_optimizers import get_optimizer
sys.path.append('../utils')
from visualization import visualize_results


class Trainer():
    def __init__(self, 
                 dataset,
                 dirs,
                 network_architecture_params,
                 augmentation_params,
                 batch_size=32,
                 epochs=50,
                 verbose=1,
                 patience=0,
                 loss_function='binary_crossentropy',
                 optimizer='Adam',
                 initial_lr=1e-4,
                 **kwargs):

        # for train & validation data
        self.dataset = dataset
        self.X_train = self.dataset.X_train
        self.y_train = self.dataset.y_train
        self.X_valid = self.dataset.X_valid
        self.y_valid = self.dataset.y_valid

        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.patience = patience
        
        self.fetch_sample_valid_data(size=5)
        self.n_batch = math.ceil(self.X_train.shape[0]/self.batch_size) 
        self.n_class = np.max(self.y_train)+1

        # for model compile
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.loss_function = loss_function

        # for model construction
        self.network_architecture_params = network_architecture_params

        # for augmentation
        self.augmentation_params = augmentation_params
        if self.augmentation_params['augmentation']:
            self.augmentor = ImageAugmentor(**self.augmentation_params)
        else:
            self.augmentor = None

        # for save results
        self.result_dir = dirs['result_dir']
        self.log_dir = dirs['log_dir']
        self.train_batch_dir = dirs['train_batch_dir']
        self.valid_batch_dir = dirs['valid_batch_dir']

        # for results logger
        self.logger = Logger()
        self.train_loss_epoch = []
        self.train_loss_batch = []
        self.valid_loss_epoch = []
        self.best_loss = float('inf')
        self.wait = 0

    def update(self):
        
        # update loop
        for idx in range(self.n_batch):
            batch_X = copy(self.X_train[idx*self.batch_size:(idx+1)*self.batch_size])
            batch_y = copy(self.y_train[idx*self.batch_size:(idx+1)*self.batch_size])

            # print(batch_X.shape)
            # print(batch_y.shape)            

            # apply transform as data augmentation
            if self.augmentor is not None:
                batch_X, batch_y = self.augmentor.augment(batch_X, batch_y, borderMode='reflect')

            #TODO: save batch_X and batch_y to self.train_batch_dir
            if idx<5 and self.current_epoch<5:
                for i in range(batch_X.shape[0]):
                    visualize_results(save_path=self.train_batch_dir+'/sample{:02d}_epoch{:04d}.png'.format((idx*batch_X.shape[0])+i, self.current_epoch),
                                      image=batch_X[i],
                                      gt=batch_y[i],
                                      pred=batch_y[i])

            # update weight
            train_loss = self.model.train_on_batch(batch_X, batch_y)

            # store loss
            self.loss_tmp.append(train_loss)
            self.train_loss_batch.append(train_loss)
            print("\rbatch: {}/{} loss: {:.5f} ".format(idx+1, self.n_batch, train_loss), end="")
    
        self.logger.plot_history(histories=[self.train_loss_batch], 
                                save_path=self.log_dir+'/loss_hist_iter.png',
                                legends=['train loss'],
                                x_label='iteration')

    def define_model(self):

        # create model
        u_net = UNet(input_shape=(self.X_train.shape[1], self.X_train.shape[2], 3), 
                     n_class=self.n_class,
                     **self.network_architecture_params)
        self.model = u_net.get_model()

        # save architecture
        self.logger.save_model_structure(self.model, 
                                        #  fig_path=self.result_dir+'/model.png',
                                         txt_path=self.result_dir+'/model.txt')
       
        # define optimizer and loss function
        self.model.compile(optimizer=get_optimizer(name=self.optimizer, lr=self.initial_lr), 
                           loss=self.loss_function)

    def train(self):

        self.define_model()

        for epoch in range(1, self.epochs+1):      
            print("-" * 80)
            print("epoch: {}/{}".format(epoch, self.epochs))
            self.current_epoch = epoch
            self.loss_tmp = []

            # shuffle
            p = np.random.permutation(self.X_train.shape[0])
            self.X_train, self.y_train = self.X_train[p], self.y_train[p]

            # update loop
            self.update()

            # summarize loss
            train_loss = np.mean(self.loss_tmp)
            val_loss = self.model.evaluate(self.X_valid, self.y_valid, batch_size=1, verbose=0)
            self.train_loss_epoch.append(train_loss)    
            self.valid_loss_epoch.append(val_loss)
            print("training loss: {:.5f}, validation loss: {:.5f}".format(train_loss, val_loss))

            # plot loss hitory
            self.logger.plot_history(histories=[self.train_loss_epoch, self.valid_loss_epoch],
                                     save_path=self.log_dir+'/loss_hist_epoch.png')

            # test using sampled validation data
            pred_sample = self.model.predict(self.X_valid_sample, batch_size=2, verbose=0)
            pred_sample = np.argmax(pred_sample, axis=3).astype(np.uint8)

            for i in range(pred_sample.shape[0]):
                visualize_results(save_path=self.valid_batch_dir+'/sample{:02d}_epoch{:04d}.png'.format(i, epoch),
                                  image=self.X_valid_sample[i],
                                  gt=self.y_valid_sample[i],
                                  pred=pred_sample[i])

            # model checkpoint
            self.current_loss = val_loss
            self.checkpoint()
            if (self.patience>0) and (self.wait>=self.patience):
                break

            self.logger.save_history_as_json({'train_loss_epoch':[str(r) for r in self.train_loss_epoch], 
                                              'train_loss_batch':[str(r) for r in self.train_loss_batch],
                                              'valid_loss_epoch':[str(r) for r in self.valid_loss_epoch]},
                                              self.log_dir+'/loss.json')


    def fetch_sample_valid_data(self, size=5):

        if size>self.X_valid.shape[0]:
            warnings.warn('indicated sample size is larger than whole validation data.\n\
                           all validation data will be used as sample data.')
            size = self.X_valid.shape[0]

        sample_idx = np.random.randint(self.X_valid.shape[0], size=size)
        self.X_valid_sample = copy(self.X_valid[sample_idx])
        self.y_valid_sample = copy(self.y_valid[sample_idx])


    def checkpoint(self):

        if self.current_loss<self.best_loss:
            print('validation loss was improved from {:.5f} to {:.5f}.'.format(self.best_loss, self.current_loss))
            self.model.save(self.result_dir+'/trained_model.h5')
            self.wait = 0
            self.best_loss = self.current_loss
        else:
            print('validation loss was not improved.')
            self.wait += 1