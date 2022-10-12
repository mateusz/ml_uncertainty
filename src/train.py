import tensorflow as tf

class Net:
    def __init__(self, ids):
        corpus = tf.data.Dataset.from_tensor_slices(ids.unique())
        vectorization = tf.keras.layers.TextVectorization(
            split=None,
            max_tokens=len(ids), # could be less
            output_mode='int',
            output_sequence_length=1,
            standardize=None
        )
        vectorization.adapt(corpus.batch(1024))

        input = tf.keras.Input(shape=(1,), dtype=tf.string, name='input')
        l = vectorization(input)
        l = tf.keras.layers.Embedding(vectorization.vocabulary_size(), 1, name='embedding')(l)
        l = tf.keras.layers.Dense(16, activation='relu', name='features')(l)
        l = tf.keras.layers.Dense(1, activation='linear', name='decision')(l)
        self.model = tf.keras.models.Model(input, outputs=l, name='Net')


def main():
    pass

if __name__ == '__main__':
    main()