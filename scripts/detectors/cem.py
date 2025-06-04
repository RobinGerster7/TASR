import numpy as np
from scripts.detectors.__base__ import Detector
import scipy.linalg
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class CEM(Detector):
    """
    Optimized Constrained Energy Minimization (CEM) detector.

    Reference:
        @article{article,
        author = {Farrand, William and Harsanyi, Joseph},
        year = {1997},
        month = {01},
        pages = {64-76},
        title = {Mapping the Distribution of Mine Tailings in the Coeur d'Alene River Valley, Idaho, Through the Use of a Constrained Energy Minimization Technique},
        volume = {59},
        journal = {Remote Sensing of Environment},
        doi = {10.1016/S0034-4257(96)00080-6}}
    """

    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Optimized forward pass for the CEM detector.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (B, C, H, W).
            target (np.ndarray): Target vector of shape (B, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (B, 1, H, W).
        """
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 supported"

        img = img[0].reshape(C, H * W).astype(np.float32)  # Shape: (C, H*W)
        target = target[0].reshape(C, 1).astype(np.float32)  # Shape: (C, 1)

        # Compute covariance matrix correctly (C, C)
        R = (img @ img.T) / (H * W)  # Shape: (C, C)

        # Add small regularization term
        lambda_identity = np.eye(C, dtype=np.float32) * 1e-6
        R_reg = R + lambda_identity  # Shape: (C, C)

        # Solve R_reg * w = target instead of computing R^(-1) * target
        w = scipy.linalg.solve(R_reg, target, assume_a="pos")  # Shape: (C, 1)

        # Compute the detection result
        result = (w.T @ img).reshape(1, 1, H, W)  # Shape: (B=1, 1, H, W)

        return result


