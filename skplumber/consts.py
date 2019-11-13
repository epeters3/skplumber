from enum import Enum

class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    @classmethod
    def has_value(cls, value):
        return value in set(item.value for item in cls)

class PrimitiveType(Enum):
    DATA_PREPARATION = 1
    PREPROCESSOR = 2
    MODEL =  3

class SKLearnAPIType(Enum):
    TRANSFORMER = 1
    PREDICTOR = 2