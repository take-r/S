from keras import optimizers

def get_optimizer(name, lr=None):
    if name == 'Adam':
        if lr is not None:
            return optimizers.Adam(lr=lr)
        else:
            return optimizers.Adam()
        
    elif name == 'SGD':
        if lr is not None:
            return optimizers.SGD(lr=lr)
        else:
            return optimizers.SGD()
        
    elif name == 'RMSprop':
        if lr is not None:
            return optimizers.RMSprop(lr=lr)
        else:
            return optimizers.RMSprop()
    
    elif name == 'Adagrad':
        if lr is not None:
            return optimizers.Adagrad(lr=lr)
        else:
            return optimizers.Adagrad()
    
    elif name == 'Adadelta':
        if lr is not None:
            return optimizers.Adadelta(lr=lr)
        else:
            return optimizers.Adadelta()
    
    elif name == 'Adamax':
        if lr is not None:
            return optimizers.Adamax(lr=lr)
        else:
            return optimizers.Adamax()
    
    elif name == 'Nadam':
        if lr is not None:
            return optimizers.Nadam(lr=lr)
        else:
            return optimizers.Nadam()
    
    else:
        raise ValueError('"{}" was not valid optimizer.'.format(name))
