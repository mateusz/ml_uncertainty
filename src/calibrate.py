import tensorflow as tf
import pandas as pd
import scipy.stats as stats

def main():
    train = pd.read_csv('data/train.csv')[['id', 'val']].sample(frac=1.0)
    val = pd.read_csv('data/val.csv')[['id', 'val']].sample(frac=1.0)
    test = pd.read_csv('data/test.csv')[['id', 'val']].sample(frac=1.0)

    model = tf.keras.models.load_model('models/best_loss', compile=False)


if __name__ == '__main__':
    main()