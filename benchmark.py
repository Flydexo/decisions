"""Run the bounded streaming pilot on public sources used by Decision Index."""
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from decisions.benchmark import source_pilot


@hydra.main(version_base="1.3", config_path="conf", config_name="benchmark")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    if not config["checkpoint"]:
        raise ValueError("Set checkpoint=/absolute/path/to/last.pt")
    source_pilot(config, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
