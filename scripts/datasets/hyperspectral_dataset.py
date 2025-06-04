import os
import numpy as np
import scipy.io as sio
import mmap


class HyperspectralImageDataset:
    """
    A dataset class for loading hyperspectral images and their ground truth maps using NumPy.

    Args:
        folder_path (str): Path to the folder containing .mat files.
        img_transforms (list, optional): List of transformation functions to apply to the hyperspectral images.
        gt_transforms (list, optional): List of transformation functions to apply to the ground truth maps.

    Expected .mat file format:
        - Each .mat file must contain two keys:
            1. **'data'** (np.ndarray): The hyperspectral image array with shape (H, W, C), where:
               - H: Height of the image
               - W: Width of the image
               - C: Number of spectral bands (channels)
            2. **'map'** (np.ndarray): The ground truth map with shape (H, W) or (H, W, 1), where:
               - H: Height of the ground truth map
               - W: Width of the ground truth map
               If the ground truth map has shape (H, W), it will be expanded to (1, H, W).

    Outputs:
        image (np.ndarray): Transformed hyperspectral image of shape (C, H, W).
        ground_truth (np.ndarray): Transformed ground truth map of shape (1, H, W).
    """

    def __init__(self, folder_path, img_transforms=None, gt_transforms=None):
        self.folder_path = folder_path
        self.file_list = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mat')
        ]
        self.img_transforms = img_transforms
        self.gt_transforms = gt_transforms

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        mat_file = self.file_list[idx]
        with open(mat_file, "rb") as f:
            mmapped_file = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            data = sio.loadmat(mmapped_file, struct_as_record=False, squeeze_me=True)

        if 'data' not in data or 'map' not in data:
            raise ValueError(f"File {mat_file} is missing 'data' or 'map'.")

        image = np.asarray(data['data'], dtype=np.float32).transpose(2, 0, 1)  # (C, H, W)
        ground_truth = np.asarray(data['map'], dtype=np.float32)

        if ground_truth.ndim == 2:
            ground_truth = np.expand_dims(ground_truth, axis=0)

        mmapped_file.close()  # Close only after loading data

        # Apply transforms
        if self.img_transforms:
            for transform in self.img_transforms:
                image = transform(image)

        if self.gt_transforms:
            for transform in self.gt_transforms:
                ground_truth = transform(ground_truth)

        return image, ground_truth



