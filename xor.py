import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import tensorflow as tf


def gaussian(mu, sigma):
    # the 0.4 scales the maximum to 1 (eyeballed)
    scaling_constant = 1 / (np.sqrt(2 * np.pi) * sigma * 0.4)
    scaling_constant = K.variable(value=scaling_constant, dtype="float32", name="scaling_constant")
    mu = K.variable(value=mu, dtype="float32", name="mu")
    sigma = K.variable(value=1/(np.sqrt(2)*sigma), dtype="float32", name="sigma")
    
    def gaussian_activation(x):
        return tf.math.multiply(
            scaling_constant,
            K.exp(
                tf.math.negative(tf.math.square(tf.math.multiply((x-mu), sigma)))
                )
            )

    return gaussian_activation


def xor_model():
    input_layer = Input(shape=(2,))
    # mean has to be positive
    output_layer = Dense(1, activation=gaussian(1, 1))(input_layer)
    model = Model(input_layer, output_layer)
    model.compile(optimizer='adam', loss="mse", metrics=["accuracy"])
    return model


def train_xor():
    # no train test split because I'm lazy
    model = xor_model()

    X = np.random.randint(0, 2, (10000, 2))
    
    y = np.empty((10000, 1))
    
    for i in range(10000):
        if X[i, 0] == X[i, 1]:
            y[i] = 0
        else:
            y[i] = 1

    model.fit(X, y, epochs=40)
    return model
    
