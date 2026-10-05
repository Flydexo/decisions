from __future__ import annotations

import os
from pathlib import Path


class Logger:
    def __init__(self, config, run_dir, resolved_config):
        self.trackio = None
        if config.get("enabled", True):
            # Explicit local logging also ignores inherited remote destination settings.
            os.environ["TRACKIO_DIR"] = str(Path(run_dir).parent / "trackio")
            remote = config.get("space_id") or config.get("server_url")
            removed = {}
            if not remote:
                for key in ("TRACKIO_SPACE_ID", "TRACKIO_SERVER_URL", "TRACKIO_DATASET_ID", "TRACKIO_BUCKET_ID",
                            "TRACKIO_REGISTRY_BUCKET_ID"):
                    if key in os.environ:
                        removed[key] = os.environ.pop(key)
            try:
                import trackio
                trackio.init(project=config["project"], name=Path(run_dir).name,
                             config=resolved_config, space_id=config.get("space_id"),
                             server_url=config.get("server_url"), auto_log_gpu=False,
                             auto_log_cpu=False, embed=False)
                self.trackio = trackio
            finally:
                os.environ.update(removed)

    def log(self, metrics, step):
        if self.trackio:
            self.trackio.log({k: v for k, v in metrics.items() if k != "step"}, step=step)

    def finish(self):
        if self.trackio:
            self.trackio.finish()
