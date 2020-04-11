from enum import Enum


class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    @classmethod
    def has_value(cls, value):
        return value in set(item.value for item in cls)


class OptimizationDirection(Enum):
    MINIMIZE = 1
    MAXIMIZE = 2


class PrimitiveType(Enum):
    PREPROCESSOR = 1
    TRANSFORMER = 2
    REGRESSOR = 3
    CLASSIFIER = 4
