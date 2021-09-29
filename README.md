


#  Generative Adversarial Networks(GAN) in TensorFlow 2.0

Tensorflow implementation of deep learning pipelines that are based on Generative Adversarial Networks.
Now it type of GAN is available such as:
* Vanilla GAN (Mnist)
* Vanilla GAN (Fashion Mnist)
* DCGAN (Planned)
* StyleGAN (Planned)


##  Prerequisites


##  Usage Example - Vanilla GAN for MNIST

 <b>Pre-trained models</b>
To test with an existing model:

    $ python3 run.py --exp=PRETRAINED  --problem=VANILLA_MNIST

 <b>Training models</b>
Or you can train by yourself

    $ python3 run.py --exp=TRAIN  --problem=VANILLA_MNIST



## Results
Due to time reasons, the following models have a low number of learning epochs. 

### Image generation
Vanilla GAN (MNIST)            |  Vanilla GAN (FASHION_MNIST)
:-------------------------:|:-------------------------:
![vanilla_mnist](./documents/VANILLA_MNIST_STEP.gif) |  ![vanilla_fashion_mnist](./documents/VANILLA_FASHION_MNIST_STEP.gif) 


### Loss
Vanilla GAN (MNIST)            |  Vanilla GAN (FASHION_MNIST)
:-------------------------:|:-------------------------:
![vanilla_mnist](./documents/VANILLA_MNIST_LOSS.gif) |  ![vanilla_fashion_mnist](./documents/VANILLA_FASHION_MNIST_LOSS.gif) 



## Monitoring model training
...WIP...

## References
1. [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
1. [Deep Convolutional Generative Adversarial Network Tutorial in TensorFlow](https://www.tensorflow.org/beta/tutorials/generative/dcgan)