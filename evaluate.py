"""Bounded streaming evaluation of a checkpoint on configured datasets."""
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json

from decisions.data import dataset_configs
from decisions.evaluation import Predictor, evaluate


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    if not config.get("resume"):
        raise ValueError("Set resume=/absolute/path/to/last.pt")
    predictor = Predictor(config["resume"], config["device"])
    report = evaluate(predictor.model, predictor.tokenizer, dataset_configs(config), predictor.device,
                      dict(config["evaluation"], preprocessing=predictor.config["model"]), config["seed"])
    path = Path(HydraConfig.get().runtime.output_dir) / "evaluation.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
