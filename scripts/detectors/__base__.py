import numpy as np
from abc import ABC, abstractmethod


class Detector(ABC):
    """
    Base class for all detectors using NumPy.

    Args:
        None
    Methods:
        forward(img, target): Abstract method to be implemented by subclasses.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Forward pass for the detector.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (B, C, H, W).
            target (np.ndarray): Target vector of shape (B, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (B, 1, H, W).
        """
        pass
