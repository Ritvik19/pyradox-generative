from pyradox_generative import ConditionalGAN
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def test_conditional_gan():
    CODINGS_SIZE = 28
    N_CHANNELS = 1
    N_CLASSES = 10
    G_INP_CHANNELS = CODINGS_SIZE + N_CLASSES
    D_INP_CHANNELS = N_CHANNELS + N_CLASSES

    (x_train, y_train), _ = keras.datasets.mnist.load_data()
    x_train = x_train[:1000]
    x_train = x_train.astype(np.float32) / 255
    x_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0
    y_train = y_train[:1000]
    y_train = keras.utils.to_categorical(y_train, 10)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset = dataset.shuffle(1024)
    dataset = dataset.batch(32, drop_remainder=True).prefetch(1)

    generator = keras.models.Sequential(
        [
            keras.Input(shape=[G_INP_CHANNELS]),
            keras.layers.Dense(7 * 7 * 3),
            keras.layers.Reshape([7, 7, 3]),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2DTranspose(
                32, kernel_size=3, strides=2, padding="same", activation="selu"
            ),
            keras.layers.Conv2DTranspose(
                1, kernel_size=3, strides=2, padding="same", activation="tanh"
            ),
        ],
        name="generator",
    )

    discriminator = keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32,
                kernel_size=3,
                strides=2,
                padding="same",
                activation=keras.layers.LeakyReLU(0.2),
                input_shape=[28, 28, D_INP_CHANNELS],
            ),
            keras.layers.Conv2D(
                3,
                kernel_size=3,
                strides=2,
                padding="same",
                activation=keras.layers.LeakyReLU(0.2),
            ),
            keras.layers.Flatten(),
            keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="discriminator",
    )

    gan = ConditionalGAN(
        discriminator=discriminator, generator=generator, latent_dim=CODINGS_SIZE
    )
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    history = gan.fit(dataset)
    return history
