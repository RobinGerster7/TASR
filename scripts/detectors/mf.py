import numpy as np
import scipy


class MF:
    """
    Matched Filter (MF) detector.

    Reference:
        @article{article,
        author = {Lockwood, Ronald and Cooley, Tracy and Jacobson, J. and Manolakis, Dimitris},
        year = {2009},
        month = {05},
        pages = {},
        title = {Is there a best hyperspectral detection algorithm?},
        volume = {7334},
        journal = {SPIE},
        doi = {10.1117/12.816917}}
    """

    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Forward pass for the MF detector.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (1, C, H, W).
            target (np.ndarray): Target vector of shape (1, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (1, 1, H, W).
        """
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 supported"

        img_reshaped = img.reshape(C, H * W)  # Reshape to (C, H*W)
        target = target.reshape(C, 1)  # Ensure target shape is (C, 1)

        # Compute mean spectrum of the background
        mean_img = np.mean(img_reshaped, axis=1, keepdims=True)

        # Compute covariance matrix with regularization
        k = np.cov(img_reshaped, rowvar=True)  # Shape: (C, C)
        k += np.eye(C) * 1e-6  # Regularization for numerical stability

        # Compute Matched Filter weight vector
        w = scipy.linalg.solve(k, target - mean_img, assume_a="pos")  # Shape: (C, 1)

        # Apply Matched Filter to each pixel
        result = (w.T @ (img_reshaped - mean_img)).reshape(1, 1, H, W)

        return result
