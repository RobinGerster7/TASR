from scripts.experiments.configs.demo_config import DemoConfig
from scripts.experiments.demo_experiment import DemoExperiment
from scripts.target_spectrum_generators.simple_target_spectrum_generators import MeanGenerator
from scripts.target_spectrum_generators.tasr import TASR
from scripts.detectors.cem import CEM
from scripts.detectors.ace import ACE
from scripts.utils.transforms import Resize, Normalize

if __name__ == "__main__":
    image_size = (100, 100)
    config = DemoConfig(
        source_folders=["datasets/SanDiego1", "datasets/SanDiego2"],
        test_folders=["datasets/SanDiego2", "datasets/SanDiego1"],
        img_transforms=[Resize(image_size, "bilinear"), Normalize()],
        gt_transforms=[Resize(image_size, "nearest")],
        detector=CEM(),
        target_spectrum_generator=TASR(),
        num_runs=1,
        plot_and_save=True
    )
    config.display()
    experiment = DemoExperiment(config)
    experiment.run()