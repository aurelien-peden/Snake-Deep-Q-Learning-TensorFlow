import tensorflow as tf
from tensorflow import keras


def make_model(input_shape, hidden_size, output_size):
    model = keras.models.Sequential([
        keras.layers.Dense(hidden_size, activation="relu",
                           input_shape=input_shape),
        keras.layers.Dense(hidden_size, activation="relu"),
        keras.layers.Dense(hidden_size, activation="relu"),
        keras.layers.Dense(output_size)
    ])

    model.build(input_shape=(None, 11))
    return model
