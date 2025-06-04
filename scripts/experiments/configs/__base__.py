from abc import ABC, abstractmethod

class BaseExperimentConfig(ABC):
    """
    Abstract base class for defining experiment configurations.
    """

    @abstractmethod
    def __init__(self) -> None:
        """
        Initializes the configuration parameters.
        """
        pass

    @abstractmethod
    def display(self) -> None:
        """
        Displays the experiment configuration in a structured format.
        """
        pass