import os
import json

def mksubdirs(config, result_dir):
    subdirs = {}
    subdirs['log_dir'] = result_dir + '/log'
    subdirs['test_dir'] = result_dir + '/test'
    subdirs['train_batch_dir'] = result_dir + '/train_batch'
    subdirs['valid_batch_dir'] = result_dir + '/valid_batch'
    # subdirs['metric_dir'] = result_dir + '/metric'
    
    for sub_dir in subdirs.values():
        os.makedirs(sub_dir, exist_ok=True)
    subdirs['result_dir'] = result_dir

    config['dirs'] = subdirs
    return config

def save_config_as_json(config, save_path):
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)