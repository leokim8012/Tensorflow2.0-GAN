import argparse
import tensorflow as tf

from datasets import dataset_factory
from models import model_factory
from trainers import vanilla_trainer

from utils import config
from utils import visualization


def run_training(input_args):

    problem = input_args.problem

    
    print('=================================================================')
    print(f'Starting pipeline for, {problem}...')
    gan_type = problem.split('_')[0]

    
    print('=================================================================')
    problem_params = config.read_config(problem)
    print(f'Loaded parameters: \n {problem_params}')

    
    print('=================================================================')
    dataset = dataset_factory.get_dataset(problem_params, problem)
    train_dataset = dataset
    print(f'Loaded dataset: \n {train_dataset}')


    
    print('=================================================================')
    print(f'Loading Models..')
    # gan_model = (generator, discriminator) = (Vanilla_Generator.VanillaGenerator(problem_params), Vanilla_Discriminator.VanillaDiscriminator(problem_params))
  
    gan_model = model_factory.model_factory(problem_params, gan_type, input_args)
    print(f'Loaded Model: {gan_model}')
    
    print('=================================================================')
    print(f'Start Training..')
    gan_model.trainer.train(train_dataset, problem_params.num_epochs)
    print(f'Finish Training')


    
    print('=================================================================')
    print(f'Saving Generator model')
    gan_model.trainer.save_model(problem)
    print(f'Done!')


def run_pretrained(input_args):
    problem = input_args.problem
    
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
    print(f'Done!')

def makeGIF(input_args):
  problem = input_args.problem
  visualization.generate_gif_from_images(
    problem,
    './outputs/png/',
    './outputs/gif/'
  )
      
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--exp',
        required=True,
        help='PRETRAINED or TRAIN',
    )

    parser.add_argument(
        '--problem',
        required=True,
        help='The GAN type',
    )
    args = parser.parse_args()

    if(args.exp == 'TRAIN'):
      run_training(args)
    elif(args.exp == 'PRETRAINED'):
      run_pretrained(args)
    elif(args.exp == 'GIF'):
      makeGIF(args)
      


if __name__ == '__main__':
    main()