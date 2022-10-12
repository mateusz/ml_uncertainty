from generators import NormalGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api

sns.set(style="whitegrid")

params = dvc.api.params_show()

g = NormalGenerator(seed=params['seed'], r=params['generate']['r'], r_loc=params['generate']['r_loc'], r_scale=params['generate']['r_scale'])

def create(name):
    d = g.generate(n=params['generate']['n_%s' % name])
    d.to_csv('data/%s.csv' % name)

    sns.boxplot(x="id", y="val", data=d, showfliers = False)
    sns.stripplot(x='id', y='val', data=d, color=".25")
    plt.savefig('evaluation/%s_stripplot.png' % name)
    plt.close()

create('train')
create('val')
create('test')