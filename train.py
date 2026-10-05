import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from decisions.trainer import train


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(config: DictConfig):
    train(config, HydraConfig.get().runtime.output_dir)


if __name__ == "__main__":
    main()
