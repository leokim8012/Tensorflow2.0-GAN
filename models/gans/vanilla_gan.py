from easydict import EasyDict as edict

from models import model
from models.gans import gan

from trainers import gan_trainer


class VanillaGAN(gan.GAN):

    def __init__(
            self,
            model_parameters: edict,
            generator: model.Model,
            discriminator: model.Model,
            gan_trainer: gan_trainer.GANTrainer
    ):
        self.num_epochs = model_parameters.num_epochs
        self._generator = generator
        self._discriminator = discriminator
        self._trainer = gan_trainer

    @property
    def generators(self):
        return [self._generator]

    @property
    def discriminators(self):
        return [self._discriminator]

    @property
    def trainer(self):
        return self._trainer
