import numpy as np
import cv2

class Resize:
    """
    Resize the spatial dimensions of a NumPy array.

    Args:
        size (tuple): Desired output size (height, width).
        interpolation (str or int): Interpolation mode as string (e.g., "bilinear", "nearest") or OpenCV constant.
            Supported values: "nearest" → cv2.INTER_NEAREST, "bilinear" → cv2.INTER_LINEAR, "bicubic" → cv2.INTER_CUBIC.

    Outputs:
        np.ndarray: Resized array of shape (C, H_out, W_out) or (1, H_out, W_out) for ground truth.
    """

    INTERPOLATION_MAP = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC
    }

    def __init__(self, size, interpolation="bilinear"):
        self.size = size  # (height, width)
        if isinstance(interpolation, str):
            if interpolation not in self.INTERPOLATION_MAP:
                raise ValueError(
                    f"Invalid interpolation mode: {interpolation}. Supported: {list(self.INTERPOLATION_MAP.keys())}")
            self.interpolation = self.INTERPOLATION_MAP[interpolation]
        else:
            self.interpolation = interpolation  # Allow direct integer values

    def __call__(self, image):
        """
        Resize the input array to the specified size.

        Args:
            image (np.ndarray): Input array of shape (C, H, W).

        Returns:
            np.ndarray: Resized array of shape (C, H_out, W_out).
        """
        return np.stack([
            cv2.resize(image[c], self.size[::-1], interpolation=self.interpolation)
            for c in range(image.shape[0])
        ], axis=0)


class Normalize:
    """
    Normalize a NumPy array to a specific range [min_val, max_val].

    Args:
        min_val (float): Minimum value of the normalized range.
        max_val (float): Maximum value of the normalized range.

    Outputs:
        np.ndarray: Normalized array of the same shape.
    """
    apply_to_gt = False

    def __init__(self, min_val=0.0, max_val=1.0):
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, image):
        image_min = np.min(image)
        image_max = np.max(image)
        if image_max == image_min:
            return np.zeros_like(image)  # Avoid division by zero
        return (image - image_min) / (image_max - image_min) * (self.max_val - self.min_val) + self.min_val


class ClampChannels:
    """
    Retain only the first `max_channels` in a hyperspectral image.

    Args:
        max_channels (int): Number of channels to retain.

    Outputs:
        np.ndarray: Clamped array of shape (max_channels, H, W).
    """
    apply_to_gt = False

    def __init__(self, max_channels):
        self.max_channels = max_channels

    def __call__(self, image):
        return image[:self.max_channels, :, :]


class ChannelSubsampling:
    """
    Subsample spectral channels by keeping every N-th channel, determined by a specified ratio.

    Args:
        keep_ratio (float): Ratio of channels to retain (e.g., 0.5 keeps roughly half the channels).
                            Must be in the range (0, 1].

    Outputs:
        np.ndarray: Subsampled array with fewer spectral channels.
    """
    apply_to_gt = False

    def __init__(self, keep_ratio: float) -> None:
        if not (0 < keep_ratio <= 1):
            raise ValueError("keep_ratio must be in the range (0, 1].")
        self.keep_ratio = keep_ratio

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply channel subsampling to the input image.

        Args:
            image (np.ndarray): Hyperspectral image of shape (C, H, W).

        Returns:
            np.ndarray: Subsampled image of shape (C_sub, H, W).
        """
        step = max(1, int(round(1 / self.keep_ratio)))
        return image[::step, :, :]
