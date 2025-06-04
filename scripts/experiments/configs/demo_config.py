from scripts.experiments.configs.__base__ import BaseExperimentConfig
from rich.console import Console
from rich.table import Table
from typing import Callable
import numpy as np

class DemoConfig(BaseExperimentConfig):
    """
    Configuration class for the DemoExperiment.

    Args:
        source_folders (list[str]): List of paths to source dataset folders.
        test_folders (list[str]): List of paths to test dataset folders.
        img_transforms (list[Callable[[np.ndarray], np.ndarray]]): List of image transforms.
        gt_transforms (list[Callable[[np.ndarray], np.ndarray]]): List of ground truth transforms.
        detector (object): Detector model instance.
        target_spectrum_generator (object): Target spectrum generator instance.
        num_runs (int): Number of experiment runs.
        plot_and_save (bool): Whether to plot and save final detection results.
    """

    def __init__(self,
                 source_folders: list[str],
                 test_folders: list[str],
                 img_transforms: list[Callable[[np.ndarray], np.ndarray]],
                 gt_transforms: list[Callable[[np.ndarray], np.ndarray]],
                 detector: object,
                 target_spectrum_generator: object,
                 num_runs: int,
                 plot_and_save: bool) -> None:
        self.img_transforms = img_transforms
        self.gt_transforms = gt_transforms
        self.detector = detector
        self.target_spectrum_generator = target_spectrum_generator
        self.num_runs = num_runs
        self.source_folders = source_folders
        self.test_folders = test_folders
        self.plot_and_save = plot_and_save


    def display(self) -> None:
        """
        Displays the experiment configuration in a formatted table.
        """
        console = Console()
        table = Table(title="\U0001F3C1 Experiment Configuration", show_header=False)
        table.add_row("\U0001F50D Detector", self.detector.__class__.__name__)
        table.add_row("\U0001F3AF Target Generator", self.target_spectrum_generator.__class__.__name__)
        table.add_row("\U0001F4C2 Source Folders", ', '.join(self.source_folders))
        table.add_row("\U0001F4C2 Test Folders", ', '.join(self.test_folders))
        table.add_row("\U0001F501 Runs", str(self.num_runs))
        table.add_row("\U0001F5BC Plot and Save", str(self.plot_and_save))
        console.print(table)
