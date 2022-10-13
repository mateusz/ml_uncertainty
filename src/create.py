from generators import NormalGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import dvc.api

sns.set(style="whitegrid")

def main():
    params = dvc.api.params_show()

    g = NormalGenerator(seed=params['seed'], r=params['generate']['r'], r_loc=params['generate']['r_loc'], r_scale=params['generate']['r_scale'])
    g.get_dists().to_csv('data/dists.csv')

    def create(name, plot=False):
        d = g.generate(n=params['generate']['n_%s' % name])
        d.to_csv('data/%s.csv' % name)

        if plot:
            sns.stripplot(x='id', y='val', data=d, palette='viridis', hue='id', legend=None)
            plt.savefig('evaluation/%s_stripplot.png' % name)
            plt.close()

    create('train', plot=True)
    create('val')
    create('test')

if __name__ == '__main__':
    main()