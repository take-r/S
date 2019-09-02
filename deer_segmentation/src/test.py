import sys
import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# import seaborn as sns
from keras.models import load_model

sys.path.append('../utils')
from calculate_metrics import evaluate
from visualization import visualize_results

# For bayesian dropout, 
# refer to: http://proceedings.mlr.press/v48/gal16.html
# Also: https://github.com/yaringal/DropoutUncertaintyExps

def save_metic_as_json(metric, save_path):
    with open(save_path, 'w') as f:
        json.dump(metric, f, indent=4, sort_keys=True)

def plot_metric(metric,
                save_path,
                ylabel=None):
    '''plot boxplot for given metrics
    '''
    #NOTE: multi-class segmentation is not assumed.

    plt.figure(figsize=(5, 10))

    data = []
    for k, v in metric.items():
        data.append(v[0])
    plt.boxplot(data)
    
    if ylabel is not None:
        plt.ylabel(ylabel)
    
    plt.tight_layout()
    plt.savefig(save_path)

def test(dataset,
         model_path,
         save_dir,
         mc_sampling,
         **kwargs):

    model = load_model(model_path)
    pred_softmax = model.predict(dataset.X_test, batch_size=2, verbose=1)
    predictions = np.argmax(pred_softmax, axis=3).astype(np.uint8)

    dice = {}
    for p, t, idx in zip(predictions,
                         np.squeeze(dataset.y_test),
                         dataset.test_idxes):
        dice[str(idx)] = evaluate(p, t) #NOTE: returned value is list-format. 

    save_metic_as_json(dice, save_dir+'/dice.json')
    plot_metric(dice, save_dir+'/boxplot_dice.png', ylabel='Dice')

    if mc_sampling:
        T = 10
        mc_predictions = []
        uncertainties = []

        # MC sampling
        for test_img in tqdm(dataset.X_test, desc='mc_sampling'):
            test_img = np.expand_dims(test_img, axis=0)
            mc_samples = np.squeeze(np.array([model.predict(test_img, batch_size=1, verbose=0) for _ in range(T)]))
            mc_predictions.append(np.argmax(np.mean(mc_samples, axis=0), axis=2).astype(np.uint8))
            uncertainties.append(np.mean(np.var(mc_samples, axis=0), axis=2))

        for idx, image, gt, pred, mc_pred, uncertainty in zip(dataset.test_idxes,
                                                              dataset.X_test,
                                                              dataset.y_test,
                                                              predictions,
                                                              mc_predictions,
                                                              uncertainties):

            # deterministic predictions
            os.makedirs(os.path.join(save_dir, 'deterministic'), exist_ok=True)
            save_path = os.path.join(save_dir, 'deterministic', '{:04d}.png'.format(idx))
            visualize_results(save_path, image, gt, pred)

            # stochastic predictions
            os.makedirs(os.path.join(save_dir, 'stochastic'), exist_ok=True)
            save_path = os.path.join(save_dir, 'stochastic', '{:04d}.png'.format(idx))
            visualize_results(save_path, image, gt, mc_pred, uncertainty=uncertainty)

    else:
        for idx, image, gt, pred in zip(dataset.test_idxes,
                                        dataset.X_test,
                                        dataset.y_test,
                                        predictions):

            # deterministic predictions
            save_path = os.path.join(save_dir, '{:04d}.png'.format(idx))
            visualize_results(save_path, image, gt, pred)

if __name__=='__main__':

    dice = {str(x):[x/10] for x in range(100) }
    plot_metric(dice, './boxplot_dice.png', ylabel='Dice')