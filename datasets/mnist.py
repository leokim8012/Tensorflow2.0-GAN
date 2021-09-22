import tensorflow as tf

from utils import data_utils


class MnistDataset():
    
    def __init__(self,model_parameters,with_labels=False,):
        self.batch_size = model_parameters.batch_size
        self.buffer_size = model_parameters.buffer_size
        if with_labels:
            self.train_dataset = self.load_data_with_labels()
        else:
            self.train_dataset = self.load_data()

    
    def __call__(self, *args, **kwargs):
        return self.train_dataset


    def load_data(self):
        (train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = data_utils.normalize_inputs(train_images)
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(self.buffer_size).batch(self.batch_size)
        return train_dataset
    
    def load_data_with_labels(self):
        (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
        train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
        train_images = data_utils.normalize_inputs(train_images)
        # Batch and shuffle the data
        train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(
            self.buffer_size).batch(
            self.batch_size)
        return train_dataset
