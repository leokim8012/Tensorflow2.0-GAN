from utils import problem_type

from datasets import fashion_mnist
from datasets import mnist


def get_dataset(input_params, dtype: problem_type.ProblemType):
    if dtype == problem_type.ProblemType.VANILLA_MNIST.name:
        print(dtype)
        return mnist.MnistDataset(input_params).load_data()

    elif dtype == problem_type.ProblemType.VANILLA_FASHION_MNIST.name:
        print(dtype)
        return fashion_mnist.FashionMnistDataset(input_params).load_data()

    else:
        raise NotImplementedError
