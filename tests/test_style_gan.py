from pyradox_generative import StyleGAN
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from functools import partial


def test_style_gan():
    gan = StyleGAN(
        target_res=32,
        start_res=4,
        filter_nums={0: 32, 1: 32, 2: 32, 3: 32, 4: 32, 5: 32},
    )
    opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

    start_res_log2 = 2
    target_res_log2 = 5

    for res_log2 in range(start_res_log2, target_res_log2 + 1):
        res = 2 ** res_log2
        for phase in ["TRANSITION", "STABLE"]:
            if res == 4 and phase == "TRANSITION":
                continue

            train_dl = create_dataloader(res)

            steps = 10

            gan.compile(
                d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                g_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
                loss_weights={"gradient_penalty": 10, "drift": 0.001},
                steps_per_epoch=steps,
                res=res,
                phase=phase,
                run_eagerly=False,
            )

            print(phase)
            history = gan.fit(train_dl, epochs=1, steps_per_epoch=steps)

    return history


def resize_image(res, image):
    # only donwsampling, so use nearest neighbor that is faster to run
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image


def create_dataloader(res):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:100, :, :]
    x_train = np.pad(x_train, [(0, 0), (2, 2), (2, 2)], mode="constant")
    x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=3), name=None)
    x_train = tf.data.Dataset.from_tensor_slices(x_train)

    batch_size = 32
    dl = x_train.map(partial(resize_image, res), num_parallel_calls=tf.data.AUTOTUNE)
    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl
