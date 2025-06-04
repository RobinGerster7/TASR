import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from scripts.experiments.__base__ import BaseExperiment
from scripts.experiments.configs.demo_config import DemoConfig
from scripts.utils import helpers
from scripts.utils.metrics import evaluation_metrics
from scripts.utils.transforms import Normalize


class DemoExperiment(BaseExperiment):
    """
    Implementation of a demo experiment using a specific detector and target spectrum generator.
    """

    def __init__(self, config: DemoConfig):
        super().__init__(config)
        self.metrics = [
            "AUC (Pf, Pd)", "AUC (τ, Pd)", "AUC (τ, Pf)", "AUC OA", "AUC SNPR", "Inference Time (s)"
        ]
        self.results = {metric: [] for metric in self.metrics}

    def run(self) -> None:
        """Executes the experiment, processes test and source images, and records performance metrics."""
        time.sleep(0.1)
        total_runs = self.config.num_runs * len(self.config.test_folders)

        with tqdm(total=total_runs, desc="Progress", bar_format="{l_bar}{bar}| ETA: {remaining}") as pbar:
            for _ in range(self.config.num_runs):
                for test_folder, source_folder in zip(self.config.test_folders, self.config.source_folders):
                    test_loader = helpers.create_dataloader(test_folder, self.config.img_transforms, self.config.gt_transforms)
                    source_loader = helpers.create_dataloader(source_folder, self.config.img_transforms, self.config.gt_transforms)

                    results = self.step_experiment(test_loader, source_loader)
                    for metric, value in results.items():
                        self.results[metric].append(value)
                    pbar.update(1)

        self.display_results()

    def step_experiment(self, test_loader: DataLoader, source_loader: DataLoader) -> dict[str, float]:
        """Runs a single iteration of the experiment, computing performance metrics."""
        start_time = time.time()
        metrics = {key: [] for key in self.metrics[:-1]}

        for (test_image, test_gt), (source_image, source_gt) in zip(test_loader, source_loader):
            source_image, source_gt, test_image = map(lambda x: x.cpu().numpy(), (source_image, source_gt, test_image))
            target, pixel_indices = self.config.target_spectrum_generator.forward(source_image, source_gt, test_image)

            detection_map = Normalize()(self.config.detector.forward(test_image, target))
            auc_metrics = evaluation_metrics(test_gt.reshape(-1), detection_map.reshape(-1))
            for key, value in zip(metrics.keys(), auc_metrics):
                metrics[key].append(value)

        averaged_metrics = {key: round(np.mean(values), 5) for key, values in metrics.items()}
        averaged_metrics["Inference Time (s)"] = round(time.time() - start_time, 2)

        if self.config.plot_and_save:
            self.plot_and_save(test_image, test_gt, detection_map, pixel_indices)

        return averaged_metrics

    def display_results(self) -> None:
        """Displays the final aggregated results in a tabular format, with standard deviation across runs."""
        console = Console()
        table = Table(title="\U0001F3C6 Overall Final Metrics", show_header=True)
        table.add_column("Metric", justify="left")
        table.add_column("Mean", justify="center")
        table.add_column("Std Dev (Across Runs)", justify="center")

        num_runs = self.config.num_runs
        num_datasets = len(self.config.test_folders)

        for metric, values in self.results.items():
            values = np.array(values).reshape(num_runs, num_datasets)  # Reshape to (runs, datasets)
            mean_per_run = np.mean(values, axis=1)  # Mean per run across datasets
            std_across_runs = np.std(mean_per_run)  # Std across runs

            table.add_row(metric, f"{np.mean(mean_per_run):.3f}", f"{std_across_runs:.3f}")

        console.print(table)


    def plot_and_save(self, image_test: np.ndarray, gt_test: np.ndarray, detection_map_test: np.ndarray,
                      pixel_indices_test: np.ndarray) -> None:
        """
        Plots and saves test images, ground truth maps, and detection maps for visualization.

        Args:
            image_test (np.ndarray): Input hyperspectral image of shape (B, C, H, W).
            gt_test (np.ndarray): Ground truth map of shape (B, 1, H, W).
            detection_map_test (np.ndarray): Detection result of shape (B, 1, H, W).
            pixel_indices_test (np.ndarray): Indices of best pixels in flattened form (N,).

        Returns:
            None: Saves images and displays plots.
        """
        save_path = "results"
        shutil.rmtree(save_path, ignore_errors=True)
        os.makedirs(save_path)

        num_channels = image_test[0].shape[0]
        indices = [num_channels // 4, num_channels // 2, 3 * num_channels // 4]
        rgb_test = helpers.convert_to_rgb(image_test, indices) if image_test is not None else None

        rows, cols = (np.unravel_index(pixel_indices_test, (image_test.shape[2], image_test.shape[3]))
                      if pixel_indices_test is not None and image_test is not None else (None, None))

        test_data = [
            ("Test Image with Best Pixels (Fake RGB)", rgb_test, f"{save_path}/test_image.jpg"),
            ("Test Ground Truth Map", gt_test.squeeze() if gt_test is not None else None,
             f"{save_path}/ground_truth.jpg"),
            ("Test Detection Map", detection_map_test.squeeze() if detection_map_test is not None else None,
             f"{save_path}/detection_map.jpg"),
        ]

        for title, data, filename in test_data:
            if data is not None:
                plt.figure(figsize=(5, 5))
                plt.imshow(data, cmap="viridis" if "Detection Map" in filename else None)
                if "test_image.jpg" in filename and rows is not None:
                    plt.scatter(cols, rows, c='red', s=25, label='Best Pixels')
                    plt.legend(loc="upper right", fontsize=12)
                plt.axis("off")
                plt.savefig(filename, bbox_inches='tight', pad_inches=0)
                plt.close()

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        for ax, (title, data, filename) in zip(axes, test_data):
            if data is not None:
                ax.imshow(data, cmap="viridis" if "Detection Map" in filename else None)
                if "test_image.jpg" in filename and rows is not None:
                    ax.scatter(cols, rows, c='red', s=25, label='Best Pixels')
                    ax.legend(loc="upper right", fontsize=12)
                ax.set_title(title, fontsize=14)
                ax.axis("off")

        plt.tight_layout()
        plt.show()

