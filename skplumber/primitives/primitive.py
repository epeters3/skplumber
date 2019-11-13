from typing import Callable, List

from skplumber.consts import ProblemType, PrimitiveType, SKLearnAPIType

class Primitive:
    def __init__(
        self,
        f: Callable,
        *,
        primitive_type: PrimitiveType,
        sklearn_api_type: SKLearnAPIType,
        supported_problem_types: List[ProblemType],

    ) -> None:
        """
        TODO: Document
        """
        self.f = f # The actual underlying sklearn primitive
        self.primitive_type = primitive_type
        self.sklearn_api_type = sklearn_api_type
        self.supported_problem_types = supported_problem_types