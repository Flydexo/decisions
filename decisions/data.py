from __future__ import annotations

import hashlib
import copy
import itertools
import json
from pathlib import Path

from datasets import IterableDataset, load_dataset

from .adapters import Adapter
from .schema import field


def load_stream(config: dict, split: str):
    # datasets may inject runtime defaults into storage_options; keep provenance immutable.
    source = copy.deepcopy(config["source"])
    if source.get("archive"):
        from .archive import archive_rows
        return IterableDataset.from_generator(archive_rows, gen_kwargs={"archive": source["archive"], "split": split})
    path = source.get("format") or source["path"]
    files = source.get("data_files")
    if files:
        files = {k: v for k, v in files.items() if k == split}
        if not files or any("REPLACE_" in str(v) for v in files.values()):
            raise ValueError(f"{config['name']}: set source.data_files.{split} to the published file; see README")
    options = {"split": split, "streaming": True}
    for key in ("name", "revision", "data_files", "features", "encoding", "encoding_errors", "delimiter", "storage_options", "filters"):
        value = files if key == "data_files" else source.get(key)
        if value is not None:
            options[key] = value
    if source.get("format") and source.get("revision"):
        options.pop("revision", None)
    return load_dataset(path, **options)


def accepted(row: dict, config: dict, role: str) -> bool:
    for rule in config.get("filters", []):
        value = field(row, rule["field"])
        if rule.get("nonempty") and (value is None or not str(value).strip()):
            return False
        if "in" in rule and value not in rule["in"]:
            return False
    holdout = config.get("holdout")
    if holdout:
        # Use stable identities (or project groups), never Python's randomized hash.
        identity = field(row, holdout["field"])
        digest = hashlib.sha256((str(holdout.get("seed", 42)) + ":" + str(identity)).encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / 2**64
        test_end = holdout.get("fraction", 0.1)
        validation_end = test_end + holdout.get("validation_fraction", 0.)
        if validation_end >= 1:
            raise ValueError("Holdout fractions must leave training rows")
        selected = (fraction < test_end if role == "eval" else
                    test_end <= fraction < validation_end if role == "validation" else
                    fraction >= validation_end)
        if not selected:
            return False
    return True


def cached_training_rows(path):
    with Path(path).open() as handle:
        for line in handle:
            yield json.loads(line)


def examples(config: dict, role: str, *, seed=42, epoch=0, shuffle_buffer=0, cache_dir=None):
    shuffle_buffer = config.get("shuffle_buffers", {}).get(role, shuffle_buffer)
    if role == "train" and cache_dir:
        import random
        stream = iter(cached_training_rows(Path(cache_dir) / f"{config['name']}.jsonl"))
        if shuffle_buffer:
            rng = random.Random(seed + epoch)
            buffer = list(itertools.islice(stream, shuffle_buffer))
            for row in stream:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = row
            rng.shuffle(buffer)
            yield from buffer
            return
        yield from stream
        return
    split = config["splits"][role]
    stream = load_stream(config, split)
    adapter = Adapter(config["schema"], stream.features)
    if shuffle_buffer:
        stream = stream.shuffle(seed=seed + epoch, buffer_size=shuffle_buffer)
    for row in stream:
        if accepted(row, config, role):
            for canonical in adapter.iter_examples(row):
                partition = config.get("validation_holdout")
                if partition and role in {"train", "validation"}:
                    # A source group keeps every expanded passage for the same
                    # query together. Otherwise hash complete model inputs.
                    identity = (field(row, partition["field"]) if partition.get("field") else
                                {"state": canonical["state"], "questions": canonical["questions"]})
                    digest = hashlib.sha256((str(partition["seed"]) + ":" +
                        json.dumps(identity, sort_keys=True, ensure_ascii=False)).encode()).digest()
                    selected_validation = int.from_bytes(digest[:8], "big") / 2**64 < partition["fraction"]
                    if (role == "validation") != selected_validation:
                        continue
                canonical["_dataset"] = config["name"]
                yield canonical


def mixed_examples(configs: list[dict], role: str, **kwargs):
    """Deterministic round robin, removing exhausted streams without materializing them."""
    active = [iter(examples(config, role, **kwargs)) for config in configs]
    while active:
        survivors = []
        for stream in active:
            try:
                yield next(stream)
                survivors.append(stream)
            except StopIteration:
                pass
        active = survivors


def sampled_examples(dataset, role, max_rows, *, seed=42, epoch=0, shuffle_buffer=0):
    """Bounded samples; class-ordered Parquet sources use label-filtered streams."""
    sampling = dataset.get("stratified_sampling")
    if not sampling:
        stream = examples(dataset, role, seed=seed, epoch=epoch, shuffle_buffer=shuffle_buffer)
        try:
            yield from itertools.islice(stream, max_rows)
        finally:
            stream.close()
        return
    values = sampling["values"]
    if not values or len(set(values)) != len(values) or max_rows < len(values):
        raise ValueError("Stratified sample needs unique classes and at least one row per class")
    if dataset["source"].get("filters"):
        raise ValueError("Combine source filters with stratification explicitly before sampling")
    base, remainder = divmod(max_rows, len(values))
    for index, value in enumerate(values):
        subset = copy.deepcopy(dataset)
        subset["source"]["filters"] = [(sampling["field"], "==", value)]
        # Each reader is released before another opens; no corpus is materialized.
        subset["shuffle_buffers"] = {role: sampling.get("shuffle_buffer", 128)}
        stream = examples(subset, role, seed=seed + index * 997, epoch=epoch,
                          shuffle_buffer=shuffle_buffer)
        quota = base + (index < remainder)
        count = 0
        try:
            for row in itertools.islice(stream, quota):
                count += 1
                yield row
        finally:
            stream.close()
        if count != quota:
            raise ValueError(f"{dataset['name']}: class {value!r} has only {count} {role} rows; expected {quota}")


def batches(stream, batch_size: int):
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    iterator = iter(stream)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def dataset_configs(config: dict) -> list[dict]:
    from omegaconf import OmegaConf
    import copy
    root = Path(__file__).resolve().parents[1] / "conf" / "dataset"
    datasets = ([OmegaConf.to_container(OmegaConf.load(root / f"{name}.yaml"), resolve=True)
                 for name in config["mixture"]] if config.get("mixture") else [copy.deepcopy(config["dataset"])])
    settings = config.get("data", {})
    if settings.get("separate_validation"):
        fraction = settings.get("validation_fraction", .1)
        if not 0 < fraction < 1:
            raise ValueError("Validation fraction must be between zero and one")
        for dataset in datasets:
            if dataset["source"].get("archive"):
                dataset["splits"]["validation"] = "validation"
            elif dataset.get("holdout"):
                dataset["splits"]["validation"] = dataset["splits"]["train"]
                dataset["holdout"]["validation_fraction"] = fraction
            else:
                if dataset["splits"]["train"] == dataset["splits"]["eval"]:
                    raise ValueError("A shared final evaluation split needs an explicit identity/group holdout")
                dataset["splits"]["validation"] = dataset["splits"]["train"]
                dataset["validation_holdout"] = {"fraction": fraction, "seed": config["seed"]}
                if dataset.get("validation_group_field"):
                    dataset["validation_holdout"]["field"] = dataset["validation_group_field"]
    overrides = settings.get("shuffle_buffers", {})
    for dataset in datasets:
        if dataset["name"] in settings.get("stratified_sampling", {}):
            dataset["stratified_sampling"] = copy.deepcopy(settings["stratified_sampling"][dataset["name"]])
        if dataset["name"] in overrides:
            dataset["shuffle_buffers"] = overrides[dataset["name"]]
        if dataset["source"].get("archive") and settings.get("archive_max_transfer_bytes"):
            dataset["source"]["archive"]["max_transfer_bytes"] = settings["archive_max_transfer_bytes"]
            if settings.get("archive_block_size_bytes"):
                dataset["source"]["archive"]["block_size_bytes"] = settings["archive_block_size_bytes"]
    return datasets


def prepare_training_cache(config: dict, datasets: list[dict], root: Path):
    """Cache a bounded training pool one source at a time, then release its reader."""
    import gc
    import resource
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    quota = config["data"]["train_pool_rows"]
    if quota < 1:
        raise ValueError("train_pool_rows must be positive")
    manifest = {"seed": config["seed"], "quota": quota, "datasets": datasets,
                "shuffle_buffer": config["training"]["shuffle_buffer"]}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())["provenance"]
        previous_sources = {d["name"]: d for d in previous["datasets"]}
        sampling_changed = any(previous[k] != manifest[k] for k in ("seed", "quota", "shuffle_buffer"))
        cached_source_changed = any(
            (root / f"{d['name']}.jsonl").exists() and previous_sources.get(d["name"]) != d
            for d in datasets)
        if sampling_changed or cached_source_changed:
            raise ValueError("Cached training samples were built with different partition or sampling settings")
    counts = {}
    for dataset in datasets:
        print("Preparing streaming training pool: " + dataset["name"], flush=True)
        path = root / f"{dataset['name']}.jsonl"
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            stream = sampled_examples(dataset, "train", quota, seed=config["seed"],
                                      shuffle_buffer=config["training"]["shuffle_buffer"])
            count = 0
            try:
                with temporary.open("w") as output:
                    for row in itertools.islice(stream, quota):
                        output.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
            finally:
                stream.close()
                gc.collect()
            if not count:
                raise ValueError(f"{dataset['name']}: no rows in its training partition")
            temporary.replace(path)
        with path.open() as handle:
            counts[dataset["name"]] = sum(1 for _ in handle)
        print(json.dumps({"cached_training_dataset": dataset["name"], "rows": counts[dataset["name"]],
                          "bytes": path.stat().st_size,
                          "max_process_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30}), flush=True)
        manifest_path.write_text(json.dumps({"provenance": manifest, "counts": counts}, indent=2))
    return counts
