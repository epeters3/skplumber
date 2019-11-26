from enum import Enum


class ProblemType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    @classmethod
    def has_value(cls, value):
        return value in set(item.value for item in cls)


class PrimitiveType(Enum):
    TRANSFORMER = 1
    REGRESSOR = 2
    CLASSIFIER = 3
