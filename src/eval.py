import models
import pandas as pd
import dvc.api
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api

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
    preds = model.predict(test['id'].sort_values().unique(), repeats=params['eval']['repeats'])

    pred_boxplots = convert_to_boxplot(preds['pred_mean'], preds['pred_std'])

    fig, ax = plt.subplots()
    sns.boxplot(data=pred_boxplots, palette='viridis', width=0.3, showfliers=False)
    sns.boxplot(x="id", y="val", data=test.sort_values('id'), 
        color='0.7',
        whis=1.5,
        showfliers=False,
        boxprops=dict(alpha=0.3),
        medianprops=dict(alpha=0.3),
        whiskerprops=dict(alpha=0.3),
        capprops=dict(alpha=0.3),
    )
    plt.title('prediction vs test')
    plt.savefig('evaluation/vs_test.png')
    plt.close()

    dists_boxplots = convert_to_boxplot(dists['loc'], dists['scale'])

    # Bad ordering
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
    plt.title('prediction vs test')
    plt.title('prediction vs underlying')
    plt.savefig('evaluation/vs_underlying.png')
    plt.close()

if __name__ == '__main__':
    main()