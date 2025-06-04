import numpy as np
from scripts.detectors.__base__ import Detector

class ACE(Detector):
    """
    Adaptive Cosine Estimator (ACE) detector.

    Reference:
        @article{article,
        author = {Manolakis, Dimitris and Marden, David and Shaw, Gary},
        year = {2003},
        month = {01},
        pages = {},
        title = {Hyperspectral image processing for automatic target detection applications},
        volume = {14},
        journal = {Lincoln Lab J}}
    """

    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Forward pass for the ACE detector.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (B, C, H, W).
            target (np.ndarray): Target vector of shape (B, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (B, 1, H, W).
        """
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 supported"
        img_flat = img.reshape(B, C, -1)  # Reshape to (B, C, H*W)
        target_flat = target.reshape(B, C, 1)  # Reshape to (B, C, 1)

        img_mean = np.mean(img_flat, axis=2, keepdims=True)  # Compute mean (B, C, 1)
        img0 = img_flat - img_mean  # Center the image

        # Compute covariance matrix R (B, C, C)
        R = np.matmul(img0, img0.transpose(0, 2, 1)) / img0.shape[2]

        # Invert covariance matrix
        R_inv = np.linalg.inv(R)

        # Compute ACE numerator and denominator using @ operator
        y0 = np.square((target_flat - img_mean).transpose(0, 2, 1) @ R_inv @ img0)
        y1 = (target_flat - img_mean).transpose(0, 2, 1) @ R_inv @ (target_flat - img_mean)
        y2 = np.sum((R_inv @ img0) * img0, axis=1, keepdims=True)

        result = y0.squeeze(1) / (y1.squeeze(1) * y2.squeeze(1))  # Shape (B, H*W)
        result = result.reshape(B, 1, H, W)  # Reshape back to (B, 1, H, W)

        return result