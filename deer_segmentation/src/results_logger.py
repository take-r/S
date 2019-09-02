import json
import os
import sys
import matplotlib.pyplot as plt
from keras.utils import plot_model

class Logger():
    def __init__(self):
        pass

    def save_prediction(self,
                        image,
                        label,
                        save_path):
        raise NotImplementedError()

    def save_history_as_json(self,
                             history,
                             save_path):

        with open(save_path, 'w') as f:       
            json.dump(history, f, indent=4, sort_keys=True)

    def plot_history(self,
                     histories,
                     save_path,
                     legends=['train loss', 
                              'valid loss'],
                     x_label='epoch',
                     y_label='loss',
                     title='model loss',
                     loc='upper right'):

        plt.figure(figsize=(12, 12))
        
        for history in histories:
            plt.plot(history)

        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(legends, loc=loc)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


    def save_model_structure(self, 
                             model,
                             fig_path=None,
                             txt_path=None):

        if fig_path:
            plot_model(model, 
                       to_file=fig_path,
                       show_shapes=True,
                       show_layer_names=True)
        
        if txt_path:
            with open(txt_path, "w") as fp:
                model.summary(print_fn=lambda x: fp.write(x + "\n"))


if __name__=='__main__':
    # sanity check

    logger = Logger()

    train_loss = [i for i in range(1000)]
    valid_loss = [i+10 for i in range(1000)]

    logger.save_history_as_json({'train_loss':train_loss}, './tmp.json')
    logger.plot_history([train_loss, valid_loss], save_path='./tmp.png')