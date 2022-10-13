import models
import pandas as pd
import dvc.api
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api
import scipy.stats as stats

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

    # Ordering and labeling still busted
    psorted = preds.sort_values('id')
    pred_boxplots = convert_to_boxplot(psorted['pred_mean'], psorted['pred_std'])

    dsorted = dists.sort_values('id')
    dists_boxplots = convert_to_boxplot(dsorted['loc'], dsorted['scale'])

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
    plt.savefig('evaluation/vs_underlying.png')
    plt.close()

    onesigma = stats.norm.cdf(1, loc=0, scale=1)-stats.norm.cdf(-1, loc=0, scale=1)
    preds = model.predict(test)
    preds['cdf'] = stats.norm.cdf(preds['val'], loc=preds['pred_mean'], scale=preds['pred_std'])
    preds['in_interval'] = (preds['cdf']>stats.norm.cdf(-1,0,1)) & (preds['cdf']<stats.norm.cdf(1,0,1))

    fig, ax = plt.subplots()
    sns.histplot(data=preds, x='cdf', hue='in_interval', element='step', bins=max(200, round(len(preds)/10)))
    plt.savefig('evaluation/cdf.png')
    plt.close()


if __name__ == '__main__':
    main()