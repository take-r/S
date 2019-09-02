import sys
import os
import random as rn
import numpy as np
import tensorflow as tf
sys.path.append('../utils')
from tf_sess import create_session, clear_session
from data_loader import DataLoader
from test import test
import argparse

def set_random_seed(seed=0):
    ''' set random seed to ensure the reproducibility.
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    rn.seed(seed)
    tf.set_random_seed(seed)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--result_dir', type=str, required=True)
    parser.add_argument('-m', '--mc_sampling', type=int, default=0)
    parser.add_argument('-s', '--seed', type=int, default=0)
    args = parser.parse_args()

    model_path = os.path.join(args.result_dir, 'trained_model.h5')
    save_dir = os.path.join(args.result_dir, 'test')
    mc_sampling = True if args.mc_sampling > 0 else False

    config = {
            'seed':args.seed,
            'data_dir':'../camvid_data',
            'data_split':'../data_index_tmp.csv',
            'preprocessing': {
                    'flag': False,
                    'normalization': False,
                    'min_max': False
                    },
            'model_path':model_path,
            'save_dir':save_dir,
            'mc_sampling':mc_sampling
    }

    # set random seed for reproducibility
    set_random_seed(seed=config['seed'])

    # load data
    dataloader = DataLoader(**config)

    # initialize training state
    create_session(gpu_id=0)

    # test
    test(dataloader, **config)

    # finish
    clear_session()

if __name__ == '__main__':    
   main()
