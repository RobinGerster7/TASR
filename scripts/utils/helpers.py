from typing import List
from scripts.utils.transforms import Resize, Normalize
from typing import Callable
import numpy as np
from torch.utils.data import DataLoader
from scripts.datasets.hyperspectral_dataset import HyperspectralImageDataset


def create_dataloader(folder_path: str,
                      img_transforms: list[Callable[[np.ndarray], np.ndarray]],
                      gt_transforms: list[Callable[[np.ndarray], np.ndarray]]) -> DataLoader:
    """
    Creates a PyTorch DataLoader for a hyperspectral dataset.

    Args:
        folder_path (str): Path to the dataset folder.
        img_transforms (list[Callable[[np.ndarray], np.ndarray]]):
            List of transformations applied to the hyperspectral image.
        gt_transforms (list[Callable[[np.ndarray], np.ndarray]]):
            List of transformations applied to the ground truth label.

    Returns:
        DataLoader: A DataLoader yielding (image, ground truth) pairs.
    """
    dataset = HyperspectralImageDataset(
        folder_path,
        img_transforms=img_transforms,
        gt_transforms=gt_transforms,
    )
    return DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)



def convert_to_rgb(image: np.ndarray, indices: List[int]) -> np.ndarray:
    """
    Converts a hyperspectral image to an RGB image using selected spectral channels.

    Args:
        image (np.ndarray): Hyperspectral image of shape (1, C, H, W),
                            where C is the number of spectral channels.
        indices (list[int]): List of three channel indices to extract for RGB visualization.
                             Should contain exactly 3 integers.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3) with values scaled to [0, 255] and dtype uint8.
    """
    assert image.ndim == 4 and image.shape[1] >= 3, "Expected input of shape (1, C, H, W) with C >= 3"
    assert len(indices) == 3, "Exactly three channel indices must be provided for RGB conversion"

    # Extract and rearrange channels to (H, W, 3)
    rgb_image = image[0, indices, :, :].transpose(1, 2, 0)

    # Normalize to range [0, 255]
    normalizer = Normalize(min_val=0, max_val=255)
    return normalizer(rgb_image).astype(np.uint8)
