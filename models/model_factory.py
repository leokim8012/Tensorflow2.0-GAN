import enum
from enum import unique

import tensorflow as tf
from easydict import EasyDict as edict

from utils import problem_type as pt


from models.gans import vanilla_gan
from models.discriminators import vanilla_discriminator
from models.generators import vanilla_generator
from trainers import vanilla_trainer



def model_factory(
        input_params: edict,
        gan_type,
        input_args,
):
    generator = generator_model_factory(input_params, input_args.problem)
    discriminator = discriminator_model_factory(input_params, input_args.problem)

    if gan_type == pt.GANType.VANILLA.name:
        generator_optimizer = tf.keras.optimizers.Adam(
          learning_rate=input_params.learning_rate_generator,
          beta_1=0.5,
        )
        discriminator_optimizer = tf.keras.optimizers.Adam(
          learning_rate=input_params.learning_rate_discriminator,
          beta_1=0.5,
        )

        gan_trainer = vanilla_trainer.VanillaGANTrainer(
          model_parameters=input_params,
          generator=generator,
          discriminator=discriminator,
          # training_name=input_params.gan_type,
          generator_optimizer=generator_optimizer,
          discriminator_optimizer=discriminator_optimizer,
        )

        return vanilla_gan.VanillaGAN(
            model_parameters=input_params,
            generator=generator,
            discriminator=discriminator,
            gan_trainer=gan_trainer,
        )
    else:
        raise NotImplementedError


def generator_model_factory(
        input_params,
        problem_type: pt.ProblemType,
):
    if problem_type == pt.ProblemType.VANILLA_MNIST.name:
        return vanilla_generator.VanillaGenerator(input_params)
    if problem_type == pt.ProblemType.VANILLA_FASHION_MNIST.name:
        return vanilla_generator.VanillaGenerator(input_params)
    else:
        raise NotImplementedError


def discriminator_model_factory(
        input_params,
        dataset_type: pt.ProblemType,
):
    if dataset_type == pt.ProblemType.VANILLA_MNIST.name:
        return vanilla_discriminator.VanillaDiscriminator(input_params)
    if dataset_type == pt.ProblemType.VANILLA_FASHION_MNIST.name:
        return vanilla_discriminator.VanillaDiscriminator(input_params)

    else:
        raise NotImplementedError
