import numpy as np
from pygad import GA
from scripts.detectors.__base__ import Detector
from scripts.detectors.cem import CEM
from scripts.utils.metrics import evaluation_metrics
from scripts.utils.transforms import Resize
from scripts.target_spectrum_generators.__base__ import TargetSpectrumGenerator
import matplotlib.pyplot as plt
import imageio
import os


class TASR(TargetSpectrumGenerator):
    """
    Test-time Adaptive Spectrum Refinement (TASR) using Genetic Algorithms.
    """

    def __init__(
        self,
        genome_length: int = 10,
        population_size: int = 30,
        generations: int = 50,
        mutation_rate: int = 25,
        tournament_size: int = 5,
        keep_parents: int = 1,
        separability_weight: float = 0.1,
        size: tuple = (100, 100),
        detector: Detector = CEM(),
    ):
        super().__init__()
        self.genome_length = genome_length
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.keep_parents = keep_parents
        self.separability_weight = separability_weight
        self.detector = detector
        self.size = size

    def seperability_score(self, target_spectrum: np.ndarray, background_spectrum: np.ndarray) -> float:
        cos_theta = np.dot(target_spectrum, background_spectrum) / (
            np.linalg.norm(target_spectrum) * np.linalg.norm(background_spectrum)
        )
        return np.arccos(np.clip(cos_theta, -1, 1))

    def fitness_function(self, ga_instance, solution: list, solution_idx: int) -> float:
        pixel_indices = np.array(solution, dtype=np.int64)
        rows, cols = pixel_indices // self.image_width, pixel_indices % self.image_width
        spectra = self.test_image[:, rows, cols]

        optimized_spectrum = np.mean(spectra, axis=1, keepdims=True).reshape(1, -1, 1, 1)

        detection_map = self.detector.forward(self.source_image.reshape(1, *self.source_image.shape),
                                              optimized_spectrum)
        detection_map_flat = detection_map.squeeze().flatten()
        ground_truth_flat = self.ground_truth.flatten()

        auc_effect, *_ = evaluation_metrics(ground_truth_flat, detection_map_flat)
        background_pixels = self.test_image.reshape(self.test_image.shape[0], -1)
        separability = self.seperability_score(optimized_spectrum.squeeze(), np.mean(background_pixels, axis=1))

        return (auc_effect + self.separability_weight * separability).item()

    def forward(
        self,
        source_image: np.ndarray,
        ground_truth: np.ndarray,
        test_image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        self.test_image_fullres = test_image.squeeze(0)  # (C, H_orig, W_orig)

        self.source_image = Resize(self.size, "bilinear")(source_image.squeeze(0))
        self.test_image = Resize(self.size, "bilinear")(self.test_image_fullres)
        self.ground_truth = Resize(self.size, "nearest")(ground_truth.squeeze(0))

        self.image_height, self.image_width = self.source_image.shape[1:]
        self.num_total_pixels = self.image_height * self.image_width

        ga_instance = GA(
            num_generations=self.generations,
            sol_per_pop=self.population_size,
            num_parents_mating=self.population_size // 2,
            fitness_func=self.fitness_function,
            num_genes=self.genome_length,
            gene_space=list(range(self.num_total_pixels)),
            mutation_percent_genes=self.mutation_rate,
            parent_selection_type="tournament",
            K_tournament=self.tournament_size,
            keep_parents=self.keep_parents,
            parallel_processing=["thread", 4],
            crossover_type="uniform",
            crossover_probability=1.0,
        )

        ga_instance.run()

        solution, fitness, _ = ga_instance.best_solution()
        pixel_indices = np.array(solution, dtype=np.int64)
        rows, cols = pixel_indices // self.image_width, pixel_indices % self.image_width
        spectra = self.test_image[:, rows, cols]
        avg_spectrum = np.mean(spectra, axis=1, keepdims=True)

        scale_h = self.test_image_fullres.shape[1] / self.image_height
        scale_w = self.test_image_fullres.shape[2] / self.image_width
        original_rows = np.clip((rows * scale_h).astype(np.int64), 0, self.test_image_fullres.shape[1] - 1)
        original_cols = np.clip((cols * scale_w).astype(np.int64), 0, self.test_image_fullres.shape[2] - 1)
        original_pixel_indices = original_rows * self.test_image_fullres.shape[2] + original_cols

        return avg_spectrum.reshape(1, -1, 1, 1), original_pixel_indices

