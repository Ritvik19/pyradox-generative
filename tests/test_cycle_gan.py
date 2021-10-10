import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from pyradox_generative import CycleGAN

tfds.disable_progress_bar()
autotune = tf.data.AUTOTUNE
orig_img_size = (286, 286)
input_img_size = (256, 256, 3)


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [*orig_img_size])
    img = tf.image.random_crop(img, size=[*input_img_size])
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label):
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img


def build_generator(name):
    return keras.models.Sequential(
        [
            keras.layers.Input(shape=input_img_size),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.Conv2D(3, 3, activation="tanh", padding="same"),
        ],
        name=name,
    )


def build_discriminator(name):
    return keras.models.Sequential(
        [
            keras.layers.Input(shape=input_img_size),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(1, 3, activation="relu", padding="same"),
        ],
        name=name,
    )


def test_cycle_gan():
    train_horses, _ = tfds.load(
        "cycle_gan/horse2zebra", with_info=True, as_supervised=True, split="trainA[:5%]"
    )
    train_zebras, _ = tfds.load(
        "cycle_gan/horse2zebra", with_info=True, as_supervised=True, split="trainB[:5%]"
    )

    buffer_size = 256
    batch_size = 1

    train_horses = (
        train_horses.map(preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )
    train_zebras = (
        train_zebras.map(preprocess_train_image, num_parallel_calls=autotune)
        .cache()
        .shuffle(buffer_size)
        .batch(batch_size)
    )

    gan = CycleGAN(
        generator_g=build_generator("gen_G"),
        generator_f=build_generator("gen_F"),
        discriminator_x=build_discriminator("disc_X"),
        discriminator_y=build_discriminator("disc_Y"),
    )

    gan.compile(
        gen_g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        gen_f_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_x_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
        disc_y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    )

    history = gan.fit(
        tf.data.Dataset.zip((train_horses, train_zebras)),
    )

    return history
