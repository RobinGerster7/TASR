import numpy as np
from scripts.target_spectrum_generators.__base__ import TargetSpectrumGenerator

class MeanGenerator(TargetSpectrumGenerator):
    """
    A class to generate the mean spectrum of the target region.
    """

    def forward(self, source_image: np.ndarray, target_map: np.ndarray, test_image: np.ndarray = None):
        """
        Generate target spectrum using the specified method.

        Args:
            source_image (np.ndarray): Input image of shape (B, C, H, W).
            target_map (np.ndarray): Target mask of shape (B, 1, H, W).
            test_image (np.ndarray): Not used in this method.

        Returns:
            tuple:
                - np.ndarray: Target spectrum of shape (B, C, 1, 1).
                - None
        """
        B, C, H, W = source_image.shape

        # Corrected transpose function
        img_flat = np.transpose(source_image, (0, 2, 3, 1)).reshape(B, -1, C)  # (B, H*W, C)
        target_flat = target_map.reshape(B, -1)  # (B, H*W)

        # Identify target indices
        mask = target_flat.astype(bool)
        ts = [img_flat[b][mask[b]] if np.any(mask[b]) else np.zeros((1, C)) for b in range(B)]  # Ensure non-empty list

        # Average target spectra for each batch
        avg_target_spectrum = np.stack(
            [t.mean(axis=0) if t.shape[0] > 0 else np.zeros(C) for t in ts], axis=0
        )  # (B, C)

        # Reshape result to (B, C, 1, 1)
        result = avg_target_spectrum[:, :, np.newaxis, np.newaxis]

        return result, None
