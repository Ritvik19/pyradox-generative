import tensorflow as tf
from tensorflow import keras

adv_loss_fn = keras.losses.MeanSquaredError()


def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


class CycleGAN(keras.Model):
    def __init__(
        self,
        generator_g,
        generator_f,
        discriminator_x,
        discriminator_y,
        lambda_cycle=10.0,
        lambda_identity=0.5,
    ):
        super().__init__()
        self.gen_G = generator_g
        self.gen_F = generator_f
        self.disc_X = discriminator_x
        self.disc_Y = discriminator_y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(
        self,
        gen_g_optimizer,
        gen_f_optimizer,
        disc_x_optimizer,
        disc_y_optimizer,
        gen_loss_fn=None,
        disc_loss_fn=None,
    ):
        super().compile()
        self.gen_g_optimizer = gen_g_optimizer
        self.gen_f_optimizer = gen_f_optimizer
        self.disc_x_optimizer = disc_x_optimizer
        self.disc_y_optimizer = disc_y_optimizer
        self.generator_loss_fn = (
            gen_loss_fn if gen_loss_fn is not None else generator_loss_fn
        )
        self.discriminator_loss_fn = (
            disc_loss_fn if disc_loss_fn is not None else discriminator_loss_fn
        )
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

    def train_step(self, batch_data):
        # x is Horse and y is zebra
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Horse to fake zebra
            fake_y = self.gen_G(real_x, training=True)
            # Zebra to fake horse -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Horse to fake zebra to fake horse): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Zebra to fake horse to fake zebra) y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_g_loss = self.generator_loss_fn(disc_fake_y)
            gen_f_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_g = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_f = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_g = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
            )
            id_loss_f = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
            )

            # Total generator loss
            total_loss_g = gen_g_loss + cycle_loss_g + id_loss_g
            total_loss_f = gen_f_loss + cycle_loss_f + id_loss_f

            # Discriminator loss
            disc_x_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_g = tape.gradient(total_loss_g, self.gen_G.trainable_variables)
        grads_f = tape.gradient(total_loss_f, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_x_grads = tape.gradient(disc_x_loss, self.disc_X.trainable_variables)
        disc_y_grads = tape.gradient(disc_y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_g_optimizer.apply_gradients(
            zip(grads_g, self.gen_G.trainable_variables)
        )
        self.gen_f_optimizer.apply_gradients(
            zip(grads_f, self.gen_F.trainable_variables)
        )

        # Update the weights of the discriminators
        self.disc_x_optimizer.apply_gradients(
            zip(disc_x_grads, self.disc_X.trainable_variables)
        )
        self.disc_y_optimizer.apply_gradients(
            zip(disc_y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_g,
            "F_loss": total_loss_f,
            "D_X_loss": disc_x_loss,
            "D_Y_loss": disc_y_loss,
        }
