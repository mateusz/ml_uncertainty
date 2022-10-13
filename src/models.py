import tensorflow as tf
import numpy as np
import scipy.stats as stats
import pandas as pd
import numpy as np
import json

class MonteCarloDropoutNet:
    def __init__(
        self,
        embed_size=1,
        dense_size=32,
        dense_dropout=0.5,
        learning_rate=0.1,
        calibration=1.0,
        model_path='models/monte_carlo_dropout',
        load=False,
    ):
        self.embed_size = embed_size
        self.dense_size = dense_size
        self.dense_dropout = dense_dropout
        self.learning_rate = learning_rate
        self.model_path = model_path
        self.calibration = calibration

        if load:
            self.model = tf.keras.models.load_model('%s' % self.model_path, compile=False)
        else:
            self.model = None

    def init(self, ids):
        corpus = tf.data.Dataset.from_tensor_slices(ids)
        vectorization = tf.keras.layers.TextVectorization(
            split=None,
            max_tokens=len(ids)+2, # could be less
            output_mode="int",
            output_sequence_length=1,
            standardize=None
        )
        vectorization.adapt(corpus.batch(1024))
        print('Vocabulary: %s' % vectorization.get_vocabulary())

        input = tf.keras.Input(shape=(1,), dtype=tf.string, name="input")
        l = vectorization(input)
        l = tf.keras.layers.Embedding(vectorization.vocabulary_size(), self.embed_size, name="embedding")(l)
        l = tf.keras.layers.Dense(self.dense_size, activation="relu", name="features")(l)
        l = tf.keras.layers.Dropout(self.dense_dropout)(l, training=True)
        l = tf.keras.layers.Dense(1, activation="linear", name="decision")(l)
        self.model = tf.keras.models.Model(input, outputs=l, name="Net")

    def train(self, X_train, y_train, X_val, y_val):
        batch_size = int(len(X_train)/50)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.MeanSquaredError()
        )

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
        mc = tf.keras.callbacks.ModelCheckpoint('%s' % self.model_path, monitor='val_loss', verbose=1, mode='min', save_best_only=True)

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=batch_size,
            callbacks=[es,mc]
        )

        self.model = tf.keras.models.load_model('%s' % self.model_path, compile=False)
        
        print('val_loss: %.4f' % np.min(history.history['val_loss']))
        with open('metrics/train.json', 'w', encoding='utf-8') as f:
            json.dump({
                'loss': np.min(history.history['loss']),
                'val_loss': np.min(history.history['val_loss']),
            }, f)

    def autotune(self, pred_mean, pred_std, true_val):
        onesigma = stats.norm.cdf(1, loc=0, scale=1)-stats.norm.cdf(-1, loc=0, scale=1)

        coverage = 0.0
        best_cal = None

        for c in np.arange(1.0, 100.0, 0.25):
            calib_std = c*pred_std

            onesigma_interval = stats.norm.interval(onesigma, pred_mean, calib_std)
            pred_low = onesigma_interval[0]
            pred_high = onesigma_interval[1]
            cdf = pd.DataFrame([true_val, pred_low, pred_high]).transpose()
            # TODO figure out whats going on in here
            cdf['in_interval'] = (cdf['val']>cdf['Unnamed 0']) & (cdf['val']<cdf['Unnamed 1'])
            in_interval = len(cdf[cdf['in_interval']])

            coverage = in_interval/len(true_val)

            if coverage>onesigma:
                best_cal = c
                break

        if best_cal is None:
            raise Exception('Failed to calibrate, retry experiment')

        return best_cal

    def calibrate(self, X_calib, y_calib):
        pred = self.predict(X_calib, calibration=1.0)
        self.calibration = self.autotune(pred['pred_mean'], pred['pred_std'], y_calib)

        print('calibration: %.2f' % self.calibration)
        calib_std = self.calibration*pred['pred_std']

        print('calib_std_mean: %.2f' % calib_std.mean())
        print('calib_std_std: %.2f' % calib_std.std())
        with open('metrics/calib.json', 'w', encoding='utf-8') as f:
            json.dump({
                'calib_std_mean': calib_std.mean(),
                'calib_std_std': calib_std.std(),
            }, f)

    def inner_predict(self, X_predict):
        return self.model.predict(X_predict, batch_size=1000)

    def predict(self, X, calibration=None, repeats=30):
        if calibration is None:
            calibration = self.calibration

        preds = pd.concat(
                [pd.DataFrame(X, columns=['id'])]*repeats
            ).reset_index()
        preds['pred'] = self.inner_predict(preds['id'].to_numpy()).reshape(len(preds))
        preds = preds.reset_index()

        predsagg = preds.groupby(['index']).mean().reset_index().rename({'pred': 'pred_mean'}, axis=1)
        predsagg['pred_std'] = calibration * preds.groupby(['index']).std().reset_index()['pred']
        predsagg = predsagg.drop(['index'], axis=1)

        return predsagg

    def save_model(self):
        with open('%s.json' % self.model_path, 'w', encoding='utf-8') as f:
            json.dump({
                'embed_size': self.embed_size,
                'dense_size': self.dense_size,
                'dense_dropout': self.dense_dropout,
                'learning_rate': self.learning_rate,
                'model_path': self.model_path,
                'calibration': self.calibration,
            }, f)
        pass

def load_model(path):
    with open('%s.json' % path, 'r', encoding='utf-8') as f:
        params = json.load(f)
        return MonteCarloDropoutNet(
            embed_size = params['embed_size'],
            dense_size = params['dense_size'],
            dense_dropout = params['dense_dropout'],
            learning_rate = params['learning_rate'],
            model_path = params['model_path'],
            calibration = params['calibration'],
            load = True
        )