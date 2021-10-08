from pyradox_generative import VAE
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def test_vae():
    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train[np.where(y_train == 0)][:100]
    x_train = x_train.astype(np.float32) / 255
    x_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0

    dataset = tf.data.Dataset.from_tensor_slices(x_train)
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(32, drop_remainder=True).prefetch(1)

    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = keras.layers.Flatten()(x)
    encoder_outputs = keras.layers.Dense(16, activation="relu")(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs, name="encoder")

    latent_inputs = keras.Input(shape=(28,))
    x = keras.layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = keras.layers.Reshape((7, 7, 64))(x)
    x = keras.layers.Conv2DTranspose(
        64, 3, activation="relu", strides=2, padding="same"
    )(x)
    x = keras.layers.Conv2DTranspose(
        32, 3, activation="relu", strides=2, padding="same"
    )(x)
    decoder_outputs = keras.layers.Conv2DTranspose(
        1, 3, activation="sigmoid", padding="same"
    )(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    vae = VAE(encoder=encoder, decoder=decoder, latent_dim=28)
    vae.compile(keras.optimizers.Adam(learning_rate=0.001))
    history = vae.fit(dataset)

    return history


test_vae()
