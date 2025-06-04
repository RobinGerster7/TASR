import numpy as np
from abc import ABC, abstractmethod

class TargetSpectrumGenerator(ABC):
    """
    Abstract base class for target spectrum generators using NumPy.

    Subclasses should implement the `forward` method to compute a target spectrum
    from the input image and target mask.

    The `forward` method should return:
        - A refined target spectrum NumPy array.
        - (Optional) The pixel indices used to compute the spectrum.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, train_image: np.ndarray, target: np.ndarray, test_image: np.ndarray):
        """
        Compute the target spectrum from the input image and target mask.

        Args:
            train_image (np.ndarray): Input image array of shape (B, C, H, W).
            target (np.ndarray): Target mask array of shape (B, 1, H, W).
            test_image (np.ndarray): Secondary image array of shape (B, C, H, W).

        Returns:
            tuple:
                - np.ndarray: Target spectrum array of shape (B, C, 1, 1).
                - (Optional) np.ndarray: Pixel indices used for spectrum computation.
        """
        pass
