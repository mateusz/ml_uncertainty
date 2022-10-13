import models
import pandas as pd
import dvc.api

def main():
    params = dvc.api.params_show()
    train = pd.read_csv('data/train.csv')[['id', 'val']].sample(frac=1.0)
    val = pd.read_csv('data/val.csv')[['id', 'val']].sample(frac=1.0)

    net = models.MonteCarloDropoutNet(
        embed_size=params['train']['embed_size'],
        dense_size=params['train']['dense_size'],
        dense_dropout=params['train']['dense_dropout'],
        learning_rate=params['train']['learning_rate'],
    )

    net.init(train['id'])
    net.train(train['id'], train['val'], val['id'], val['val'])
    net.calibrate(val['id'], val['val'])
    net.save_model()

if __name__ == '__main__':
    main()