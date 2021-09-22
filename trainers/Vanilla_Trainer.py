import tensorflow as tf

from layers import losses
from models import model
from trainers import Gan_Trainer


import numpy as np
from matplotlib import pyplot as plt


class VanillaGANTrainer(Gan_Trainer.GANTrainer):

    def __init__(
            self,
            model_parameters,
            generator: model.Model,
            discriminator: model.Model,
            generator_optimizer,
            discriminator_optimizer,
            callbacks=None,
    ):
        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer

        self.latent_size = model_parameters.latent_size
        super().__init__(
            model_parameters=model_parameters,

            generators={'generator': generator},
            discriminators={'discriminator': discriminator},

            generators_optimizers={
                'generator_optimizer': self.generator_optimizer
            },
            discriminators_optimizers={
                'discriminator_optimizer': self.discriminator_optimizer
            },

            callbacks=callbacks,
        )

    @tf.function
    def train_step(self, batch):
        real_examples = batch
        generator_inputs = tf.random.normal([self.batch_size, self.latent_size])

        with tf.GradientTape(persistent=True) as tape:
            fake_examples = self.generator(generator_inputs, training=True)
            real_output = self.discriminator(real_examples, training=True)
            fake_output = self.discriminator(fake_examples, training=True)

            generator_loss = losses.generator_loss(fake_output)
            discriminator_loss = losses.discriminator_loss(real_output, fake_output)

        gradients_of_generator = tape.gradient(
            target=generator_loss,
            sources=self.generator.trainable_variables,
        )
        gradients_of_discriminator = tape.gradient(
            target=discriminator_loss,
            sources=self.discriminator.trainable_variables,
        )

        self.generator_optimizer.apply_gradients(
            grads_and_vars=zip(gradients_of_generator, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            grads_and_vars=zip(gradients_of_discriminator, self.discriminator.trainable_variables)
        )

        return {
            'generator_loss':     generator_loss,
            'discriminator_loss': discriminator_loss
        }
