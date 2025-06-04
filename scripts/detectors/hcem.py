import numpy as np


class HCEM:
    """
    Hierarchical Constrained Energy Minimization (HCEM) detector.

    Reference:
        @ARTICLE{7175025,
        author={Zou, Zhengxia and Shi, Zhenwei},
        journal={IEEE Transactions on Geoscience and Remote Sensing},
        title={Hierarchical Suppression Method for Hyperspectral Target Detection},
        year={2016},
        volume={54},
        number={1},
        pages={330-342},
        keywords={Detectors;Hyperspectral imaging;Object detection;Algorithm design and analysis;Correlation;Convergence;Constrained energy minimization (CEM);hierarchical structure;hyperspectral target detection;nonlinear suppression function;Constrained energy minimization (CEM);hierarchical structure;hyperspectral target detection;nonlinear suppression function},
        doi={10.1109/TGRS.2015.2456957}}
    """

    def __init__(self, lambda_: float = 200, epsilon: float = 1e-6, max_iter: int = 100):
        """
        Initialize the HCEM detector with hyperparameters.

        Args:
            lambda_ (float): Weighting factor for hierarchical filtering.
            epsilon (float): Convergence threshold.
            max_iter (int): Maximum number of iterations.
        """
        self.lambda_ = lambda_
        self.epsilon = epsilon
        self.max_iter = max_iter

    def forward(self, img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Forward pass for the HCEM detector.

        Args:
            img (np.ndarray): Input hyperspectral image of shape (1, C, H, W).
            target (np.ndarray): Target spectrum of shape (1, C, 1, 1).

        Returns:
            np.ndarray: Detection result of shape (1, 1, H, W).
        """
        B, C, H, W = img.shape
        assert B == 1, "Only B=1 supported"

        # Reshape image to (C, N) where N = H * W (flattened pixels)
        X = img.reshape(C, -1)
        N = X.shape[1]  # Number of pixels

        # Reshape target to (C, 1)
        d = target.reshape(C, 1)

        # Initialize weight vector and energy tracking
        Weight = np.ones((1, N))
        y_old = np.ones((1, N))
        Energy = []

        for T in range(self.max_iter):
            # Apply hierarchical weighting to each pixel
            X_weighted = X * Weight

            # Compute covariance matrix
            R = (X_weighted @ X_weighted.T) / N

            # Add small regularization for stability
            R_inv = np.linalg.inv(R + 1e-4 * np.eye(C))

            # Compute the HCEM filter
            w = R_inv @ d / (d.T @ R_inv @ d)

            # Compute detection result
            y = (w.T @ X).reshape(1, N)

            # Update weight vector
            Weight = 1 - np.exp(-self.lambda_ * y)
            Weight = np.maximum(Weight, 0)  # Ensure non-negative weights

            # Convergence check
            res = np.linalg.norm(y_old) ** 2 / N - np.linalg.norm(y) ** 2 / N
            Energy.append(np.linalg.norm(y) ** 2 / N)
            y_old = y

            if abs(res) < self.epsilon:
                break

        # Reshape result back to (1, 1, H, W)
        result = y.reshape(1, 1, H, W)

        return result
