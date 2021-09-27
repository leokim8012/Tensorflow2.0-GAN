import enum


class ProblemType(enum.Enum):
    VANILLA_MNIST = 0,
    VANILLA_FASHION_MNIST = 1


class GANType(enum.Enum):
    VANILLA = 'vanilla',


def model_type_values():
    return [i.name for i in GANType]



def dataset_type_values():
    return [i.name for i in ProblemType]
