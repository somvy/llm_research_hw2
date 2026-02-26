from abc import ABC, abstractmethod
from typing import Optional

from base.data import Data
from base.verifier import Verifier


class Env(ABC):
    def __init__(self, name: str, verifier: Verifier):
        self.name = name
        self.verifier = verifier()

    @abstractmethod
    def generate(self, num_of_questions: int = 100, max_attempts: int = 100,
                 difficulty: Optional[int] = 1):
        raise NotImplementedError

    def verify(self, data: Data, test_solution: str):
        return self.verifier.verify(data, test_solution)

    @abstractmethod
    def extract_answer(self, test_solution: str):
        raise NotImplementedError
