class ConfigSetter():
    def __init__(self,
                 data_dir,
                 batch_size,
                 verbose,
                 epochs,
                 patience,
                 initial_lr,
                 val_ratio,
                 loss,
                 optimizer,
                 calc_uncertainty,
                 normalize,
                 noise_island_ratio
                 ):

        raise NotImplementedError()


    def set_architecture_params(self):
        raise NotImplementedError()

    def set_augmentation_params(self):
        raise NotImplementedError()
    
    def make_subdirs(self):
        raise NotImplementedError()