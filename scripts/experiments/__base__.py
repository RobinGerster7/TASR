import os
from abc import ABC, abstractmethod
import numpy as np
from rich.console import Console
from rich.table import Table
from scripts.experiments.configs.__base__ import BaseExperimentConfig


class BaseExperiment(ABC):
    """
    Abstract base class for defining an experiment.

    Args:
        config (BaseExperimentConfig): Experiment configuration instance.
    """

    def __init__(self, config: BaseExperimentConfig) -> None:
        self.config = config

    @abstractmethod
    def run(self) -> None:
        """
        Executes the experiment. Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def display_results(self) -> None:
        """
        Displays the final computed metrics in a formatted table.
        """
        pass
