import os
import tensorflow as tf


from abc import abstractmethod
from typing import List

from tqdm import tqdm

from callbacks import basic_callbacks
from callbacks import callback

from datasets import abstract_dataset
from models import model


from tensorflow.keras.optimizers import Optimizer as optimizers

from utils import visualization

class GANTrainer:
    def __init__(
            self,
            model_parameters,
            
            generators: List[model.Model],
            discriminators: List[model.Model],

            generators_optimizers: List[optimizers],
            discriminators_optimizers: List[optimizers],

            callbacks: List[callback.Callback] = None,
    ):
        self.model_parameters = model_parameters
        self.batch_size = model_parameters.batch_size
        self.generators = generators
        self.discriminators = discriminators

        self.global_step = 0
        self.epoch = 0

        self.generators_optimizers = generators_optimizers
        self.discriminators_optimizers = discriminators_optimizers

        self.SEED = tf.random.normal([model_parameters.num_examples_to_generate, model_parameters.latent_size])
        self.save_images_every_n_steps = model_parameters.save_images_every_n_steps


        default_callbacks = [
            basic_callbacks.GlobalStepIncrementer(),
        ]
        self.callbacks = default_callbacks

    @abstractmethod
    def train_step(self, batch):
        raise NotImplementedError

    def train(self, dataset: abstract_dataset.Dataset, num_epochs: int,):
        global_step = 0
        epoch_tqdm = tqdm(iterable=range(num_epochs),desc="Epochs")

        g_loss = []
        d_loss = []
        step_loss =[]

        for self.epoch in epoch_tqdm:
            self.on_epoch_begin()

            visualization.generate_and_save_images(
                self.generator, 
                self.generator.model_name + '-Epoch-' + str(self.epoch + 1), 
                self.SEED
            )

            dataset_tqdm = tqdm(iterable=dataset,desc="Batches",leave=True)
            for batch in dataset_tqdm:
                self.on_training_step_begin()
                losses = self.train_step(batch)
                self.on_training_step_end()     

                if(self.global_step % self.save_images_every_n_steps == 0):
                    visualization.generate_and_save_images(
                        self.generator, 
                        self.generator.model_name + '-Step-' + str(self.global_step), 
                        self.SEED
                    )
                    g_loss.append(losses['generator_loss'].numpy())
                    d_loss.append(losses['discriminator_loss'].numpy())
                    step_loss.append(self.global_step)

                    visualization.save_loss_image(
                        g_loss, 
                        d_loss,
                        self.generator.model_name + '-Losses-' + str(self.global_step),
                        step_loss
                        )
                postfix = 'Step: ' + str(self.global_step) + ' | Generator Loss: ' + str(losses['generator_loss'].numpy()) + ' | Discriminator Loss: ' + str(losses['discriminator_loss'].numpy())
                dataset_tqdm.set_postfix_str(postfix)
                dataset_tqdm.refresh()

            self.on_epoch_end()

    def on_epoch_begin(self):
        for c in self.callbacks:
            c.on_epoch_begin(self)

    def on_epoch_end(self):
        for c in self.callbacks:
            c.on_epoch_end(self)

    def on_training_step_begin(self):
        for c in self.callbacks:
            c.on_training_step_begin(self)

    def on_training_step_end(self):
        for c in self.callbacks:
            c.on_training_step_end(self)


    def save_model(self, name):
        path=os.getcwd() + '/pretrained-models/' 
        self.generator.model.save(path + name + '.h5')