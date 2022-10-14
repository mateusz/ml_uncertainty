import models
import pandas as pd
import dvc.api
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api
import scipy.stats as stats
from tensorflow.keras.losses import MeanSquaredError
import json

def convert_to_boxplot(means, stds):
    q1 = stats.norm.ppf(0.25, loc=means, scale=stds)
    median = stats.norm.ppf(0.5, loc=means, scale=stds)
    q3 = stats.norm.ppf(0.75, loc=means, scale=stds)
    iqr = q3-q1
    w_low = q1 - 1.5*iqr
    w_high = q3 + 1.5*iqr

    return np.array([w_low, q1, median, q3, w_high])

def main():
    params = dvc.api.params_show()
    test = pd.read_csv('data/test.csv')[['id', 'val']].sample(frac=1.0)
    dists = pd.read_csv('data/dists.csv')

    model = models.load_model('models/monte_carlo_dropout')
    d = pd.DataFrame(test['id'].unique(), columns=['id'])
    preds = model.predict(d['id'], repeats=params['eval']['repeats'])

    preds = preds.set_index('id')
    dists = dists.set_index('id')

    # Force into the same label
    dists['pred_mean'] = preds['pred_mean']
    dists['pred_std'] = preds['pred_std']

    mse = MeanSquaredError()
    underlying_loc_mse = mse(dists['loc'], dists['pred_mean']).numpy()
    underlying_scale_mse = mse(dists['scale'], dists['pred_std']).numpy()

    pred_boxplots = convert_to_boxplot(dists['pred_mean'], dists['pred_std'])
    dists_boxplots = convert_to_boxplot(dists['loc'], dists['scale'])

    fig, ax = plt.subplots()
    sns.boxplot(data=pred_boxplots, palette='viridis', width=0.3, showfliers=False)
    sns.boxplot(data=dists_boxplots,
        color='0.7',
        showfliers=False,
        boxprops=dict(alpha=0.3),
        medianprops=dict(alpha=0.3),
        whiskerprops=dict(alpha=0.3),
        capprops=dict(alpha=0.3),
    )
    plt.title('prediction vs underlying')
    plt.xlabel('id')
    plt.ylabel('val')
    plt.savefig('evaluation/vs_underlying.png')
    plt.close()

    onesigma = stats.norm.cdf(1, loc=0, scale=1)-stats.norm.cdf(-1, loc=0, scale=1)
    preds = model.predict(test)
    preds['cdf'] = stats.norm.cdf(preds['val'], loc=preds['pred_mean'], scale=preds['pred_std'])
    preds['in_onesigma'] = (preds['cdf']>stats.norm.cdf(-1,0,1)) & (preds['cdf']<stats.norm.cdf(1,0,1))

    fig, ax = plt.subplots()
    sns.histplot(data=preds, x='cdf', hue='in_onesigma', element='step', bins=max(200, round(len(preds)/10)))
    plt.title('CDF distribution (sample counts at given CDF value)')
    plt.savefig('evaluation/cdf_dist.png')
    plt.close()

    preds = preds.set_index('id')
    mse = MeanSquaredError()
    preds = preds[['pred_std']].join(dists[['scale']])
    test_std_mse = mse(preds['scale'], preds['pred_std']).numpy()
    print('test_std_mse: %.4f' % test_std_mse)
    print('underlying_loc_mse: %.4f' % underlying_loc_mse)
    print('underlying_scale_mse: %.4f' % underlying_scale_mse)
    with open('metrics/eval.json', 'w', encoding='utf-8') as f:
        json.dump({
            'test_std_mse': float(test_std_mse),
            'underlying_loc_mse': float(underlying_loc_mse),
            'underlying_scale_mse': float(underlying_scale_mse),
        }, f)



if __name__ == '__main__':
    main()