from abc import ABC, abstractmethod
from typing import Dict


class LLMProvider(ABC):
    """
    Abstract LLM provider interface.
    """

    @abstractmethod
    def generate(self, *, system: str, user: str) -> str:
        """
        Generate raw text response from LLM.
        Must return a STRING (JSON expected).
        """
        raise NotImplementedError
