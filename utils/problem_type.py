import enum


class ProblemType(enum.Enum):
    VANILLA_MNIST = 0,
    VANILLA_FASHION_MNIST = 1

def dataset_type_values():
    return [i.name for i in ProblemType]
