import argparse
import tensorflow as tf

from datasets import mnist

from models.generators import Vanilla_Generator
from models.discriminators import Vanilla_Discriminator

from trainers import Vanilla_Trainer

from utils import config
from utils import visualization


def run_training(input_args):

    problem = input_args.gan_type

    
    print('=================================================================')
    print(f'Starting pipeline for, {problem}...')
    gan_type = problem.split('_')[0]

    
    print('=================================================================')
    problem_params = config.read_config(problem)
    print(f'Loaded parameters: \n {problem_params}')

    
    print('=================================================================')
    database = mnist.MnistDataset(problem_params)
    train_dataset = database.load_data()
    print(f'Loaded dataset: \n {train_dataset}')


    
    print('=================================================================')
    print(f'Loading Models..')
    gan_model = (generator, discriminator) = (Vanilla_Generator.VanillaGenerator(problem_params), Vanilla_Discriminator.VanillaDiscriminator(problem_params))
    print(f'Loaded Generatpr: {gan_model[0]}')
    print(f'Loaded Discriminator: {gan_model[1]}')


    
    print('=================================================================')
    print(f'Loading Trainer..')

    generator_optimizer = tf.keras.optimizers.Adam(
      learning_rate=problem_params.learning_rate_generator,
      beta_1=0.5,
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
      learning_rate=problem_params.learning_rate_discriminator,
      beta_1=0.5,
    )

    gan_trainer = Vanilla_Trainer.VanillaGANTrainer(
      model_parameters=problem_params,
      generator=gan_model[0],
      discriminator=gan_model[1],
      # training_name=input_args.gan_type,
      generator_optimizer=generator_optimizer,
      discriminator_optimizer=discriminator_optimizer,
    )
    print(f'Loaded Trainer: {gan_trainer}')


    
    print('=================================================================')
    print(f'Start Training..')
    gan_trainer.train(train_dataset, problem_params.num_epochs)
    print(f'Finish Training')


    
    print('=================================================================')
    print(f'Saving Generator model')
    gan_trainer.save_model(problem)
    print(f'Done!')


def run_pretrained(input_args):
    problem = input_args.gan_type

    
    print('=================================================================')
    print(f'Starting pipeline for, {problem}...')
    gan_type = problem.split('_')[0]

    
    print('=================================================================')
    problem_params = config.read_config(problem)
    print(f'Loaded parameters: \n {problem_params}')

    print(f'Loading Model..')
    gan_model = tf.keras.models.load_model('./pretrained-models/' + problem +'.h5')
    gan_model.summary() 

    print('=================================================================')
    print(f'Start Testing Model..')
    test_input = tf.random.normal([problem_params.num_examples_to_generate, problem_params.latent_size])
    visualization.generate_and_save_images(
      gan_model, 
      problem + '_TEST', 
      test_input,
      './pretrained-outputs/'
    )


    print(f'Making GIF..')
    visualization.generate_gif_from_images(
      problem,
      './outputs/png/',
      './outputs/gif/'
    )
    print(f'Done!')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        required=True,
        help='PRETRAINED or TRAIN',
    )

    parser.add_argument(
        '--gan_type',
        required=True,
        help='The GAN type',
    )
    args = parser.parse_args()

    if(args.exp == 'TRAIN'):
      run_training(args)
    elif(args.exp == 'PRETRAINED'):
      run_pretrained(args)
      


if __name__ == '__main__':
    main()