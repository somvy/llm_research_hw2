from abc import ABC, abstractmethod

from base.data import Data


class Verifier(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def verify(self, data: Data, test_answer: str):
        raise NotImplementedError

    @abstractmethod
    def extract_answer(self, test_solution: str):
        raise NotImplementedError
