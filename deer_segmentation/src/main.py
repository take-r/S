import sys
import os
import random as rn
from datetime import datetime
import numpy as np
from glob import glob

import tensorflow as tf

sys.path.append('../utils')
from tf_sess import create_session, clear_session
from config_util import save_config_as_json, mksubdirs

from data_loader import DataLoader
from model_trainer import Trainer
from test import test

def set_random_seed(seed=0):
    ''' set random seed to ensure the reproducibility.
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)

#TODO: use argparser
def main():
    config = {
            'seed':0,
            'data_dir':'../camvid_data',
            'data_split':'../data_index_tmp.csv',
            'verbose': 1,
            'batch_size': 1,
            'epochs': 100,
            'patience': 20,
            'initial_lr': 1e-4,
            'loss_function': 'sparse_categorical_crossentropy',
            'optimizer': 'Adam',
            'mc_sampling': False,
            'network_architecture_params':{
                    'n_kernel': 32,
                    'n_layer': 5,
                    'batch_norm': True,
                    'bayesian_state':True,
                    'encoder_drop_rate': 0.2,
                    'decoder_drop_rate': 0.2,
                    'activation': 'relu',
                    'kernel_initializer': 'he_normal',
                    'kernel_size': 3,
                    'pool_size': 2,
                    'last_activation':'softmax',
                    'upsample_method':'upsampling',
                    'residual':False
                    },
            'augmentation_params': {
                'augmentation': True,
                'seed':0,
                'horizontal_flip':True,
                'zoom_range':0.2,
                'width_shift_range':0.1,
                'height_shift_range':0.1,
                'shear_range':8, # [deg]
                'rotation_range':10, # [deg]
                'elastic_transform_flag':False,
                'elastic_transform_alpha':2,
                'elastic_transform_sigma':.8,
                'random_erasing_flag':True,
                'random_erasing_size':(0.02, 0.05),
                'random_erasing_ratio':(0.3, 1.0)
                    },
            'preprocessing': {
                    'flag': False,
                    'normalization': False,
                    'min_max': False
                    },
            'postprocessing': {
                    'flag': True,
                    'island_ratio': 0.05
                    }
            }

    # set random seed for reproducibility
    set_random_seed(seed=config['seed'])

    # timepoint
    today = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = '../results/{}'.format(today)
    os.makedirs(result_dir, exist_ok=True)

    # save config
    config = mksubdirs(config, result_dir)
    save_config_as_json(config, result_dir+'/config.json')

    # load data
    dataloader = DataLoader(**config)

    # initialize training state
    create_session(gpu_id=0)

    # train
    trainer = Trainer(dataloader, **config)
    trainer.train()

    # test
    test(dataloader,
         model_path=config['dirs']['result_dir']+'/trained_model.h5',
         save_dir=config['dirs']['test_dir'],
         mc_sampling=config['mc_sampling'])

    # finish
    clear_session()

if __name__ == '__main__':
   main()
