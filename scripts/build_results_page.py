"""Build an offline results page from saved measurements; never launch training."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
NAMES = {
    "ag_news": "AG News", "boolq": "BoolQ", "sst5": "SST-5",
    "arc_challenge": "ARC-Challenge", "banking77": "Banking77",
    "mnli": "MultiNLI", "yelp_review_full": "Yelp Review Full", "trec": "TREC",
    "dbpedia14": "DBpedia 14", "amazon_reviews_multi_en": "Amazon Reviews · EN",
    "imdb": "IMDb", "openbookqa": "OpenBookQA", "commonsenseqa": "CommonsenseQA",
    "aegis": "Aegis Safety 2.0", "consumer_finance": "Consumer Finance",
    "codereviewer": "CodeReviewer", "flakeflagger": "FlakeFlagger",
}
ABLATIONS = {
    "baseline": ("Sampled reward", "Full decision head; log + spherical + ordinal RPS rewards."),
    "cross_entropy": ("Cross-entropy", "Same head and data; supervised objective replaces sampled reward."),
    "no_question_type": ("No question type", "Remove the question-type embedding."),
    "no_transformer": ("No transformer", "Remove the two trainable transformer layers."),
    "linear_scorer": ("Linear scorer", "Replace the scoring network with a linear layer."),
    "no_spherical": ("No spherical reward", "Set the spherical reward weight to zero."),
    "no_rps": ("No ordinal RPS", "Set the ranked probability score reward weight to zero."),
}


def read_json(path: Path):
    return json.loads(path.read_text())


def large_run():
    run = ROOT / 'outputs/all_large_8h'
    if not (run / 'status.json').exists():
        return None
    metrics = []
    path = run / 'metrics.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # A concurrent writer may not have finished the last line.
    history = run / 'validation_history.jsonl'
    validations = [json.loads(line) for line in history.read_text().splitlines() if line] if history.exists() else []
    manifest = run / 'training_samples/manifest.json'
    pools = read_json(manifest)['counts'] if manifest.exists() else {}
    summary_path = run / 'final_evaluation.json'
    benchmark_path = run / 'benchmark/summary.json'
    correction_path = REPORTS / 'sampling_correction.json'
    # Bound SVG complexity; exact per-update measurements remain in the original log.
    stride = max(1, (len(metrics) + 299) // 300)
    plot = metrics[::stride]
    if metrics and (not plot or plot[-1] != metrics[-1]):
        plot.append(metrics[-1])
    return {'status':read_json(run / 'status.json'), 'pool_counts':pools,
            'latest':metrics[-1] if metrics else None, 'metrics':plot,
            'peak_driver_gib':max((m.get('memory/driver_gib',0) for m in metrics),default=None),
            'peak_process_rss_gib':max((m.get('memory/process_peak_rss_gib',0) for m in metrics),default=None),
            'validation_history':validations,
            'summary':read_json(summary_path) if summary_path.exists() else None,
            'sampling_correction':read_json(correction_path) if correction_path.exists() else None,
            'audit':read_json(REPORTS / 'all_large_audit.json') if (REPORTS / 'all_large_audit.json').exists() else None,
            'benchmark':read_json(benchmark_path) if benchmark_path.exists() else None,
            'page_built_at':datetime.now(timezone.utc).isoformat()}


def main():
    training = read_json(REPORTS / "training_summary.json")
    benchmark = read_json(REPORTS / "benchmark_summary.json")
    schemas = read_json(REPORTS / "schema_validation.json")
    runs = []
    for key, variant in [("pilot_streaming", "baseline"),
                         ("pilot_streaming_cross_entropy", "cross_entropy")]:
        metrics = [json.loads(line) for line in
                   (ROOT / "outputs" / key / "metrics.jsonl").read_text().splitlines() if line]
        assert len(metrics) == training[key]["steps"]
        runs.append({"id": variant, "name": ABLATIONS[variant][0],
                     "summary": training[key], "metrics": metrics,
                     "path": f"outputs/{key}"})
    datasets = []
    for key, validation in schemas.items():
        config = yaml.safe_load((ROOT / "conf" / "dataset" / f"{key}.yaml").read_text())
        source = config["source"]
        if source.get("path"):
            url = "https://huggingface.co/datasets/" + source["path"]
            source_label = source["path"]
        elif key in {"codereviewer", "flakeflagger"}:
            url = f"https://zenodo.org/records/{'6900648' if key == 'codereviewer' else '5014076'}"
            source_label = "Zenodo · " + ("6900648" if key == "codereviewer" else "5014076")
        else:
            url, source_label = "https://huggingface.co/datasets/CogComp/trec", "CogComp/trec · pinned parquet"
        datasets.append({"id": key, "name": NAMES[key], "validation": validation,
                         "trained": key in runs[0]["summary"]["shuffled_evaluation"],
                         "source": source_label, "url": url,
                         "schema": config["schema"], "splits": config["splits"],
                         "holdout": config.get("holdout"),
                         "transport": "HTTP range ZIP" if "archive" in source else "HF streaming",
                         "revision": source.get("revision")})
    benchmark_rows = [json.loads(line) for line in
                      (ROOT / "outputs/benchmark_streaming_pilot/results.jsonl").read_text().splitlines() if line]
    for family, stats in benchmark["sources"].items():
        supported = [row for row in benchmark_rows if row["family"] == family and row["status"] == "ok"]
        stats["uniform_chance"] = (sum(1 / len(row["payload"]["questions"]["q1"]["criteria"])
                                        for row in supported) / len(supported)) if supported else None
    large = large_run()
    collapse_path = REPORTS / 'collapse_rlcd_pilot_summary.json'
    data = {"names": NAMES, "runs": runs, "training": training, "benchmark": benchmark, 'large':large,
            'collapse_pilot':read_json(collapse_path) if collapse_path.exists() else None,
            "datasets": datasets, "ablations": [
                {"id": key, "name": name, "description": description,
                 "measured": key in {run["id"] for run in runs},
                 "config": yaml.safe_load((ROOT / "conf/ablation" / f"{key}.yaml").read_text())}
                for key, (name, description) in ABLATIONS.items()],
            "source_files": ["reports/training_summary.json", "reports/benchmark_summary.json",
                             "reports/schema_validation.json", "outputs/pilot_streaming/metrics.jsonl",
                             "outputs/pilot_streaming_cross_entropy/metrics.jsonl",
                             "outputs/benchmark_streaming_pilot/results.jsonl"]}
    # Escape HTML-significant characters so arbitrary dataset/config text cannot close the script element.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")
    template = (REPORTS / "results.template.html").read_text()
    assert template.count("__REPORT_DATA__") == 1
    destination = REPORTS / "results.html"
    destination.write_text(template.replace("__REPORT_DATA__", payload))
    (REPORTS / "pilot-data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    if large:
        (REPORTS / 'all-large-data.json').write_text(json.dumps(large,ensure_ascii=False,indent=2,allow_nan=False))
        with (REPORTS / 'all-large-evaluation.csv').open('w',newline='') as output:
            writer=csv.writer(output)
            fields=['rows','questions','accuracy','nll','brier','entropy_confidence','probability_ece','unsupported_rows']
            writer.writerow(['dataset',*fields])
            if large['summary']:
                for name, scores in large['summary']['final_evaluation'].items():
                    writer.writerow([name,*(scores.get(k) for k in fields)])
    fields = ["accuracy", "nll", "brier", "entropy_confidence", "probability_ece"]
    with (REPORTS / "holdout-ablations.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["dataset", "objective", "rows", "questions", *fields])
        for run in runs:
            for key, scores in run["summary"]["shuffled_evaluation"].items():
                writer.writerow([key, run["id"], scores["rows"], scores["questions"],
                                 *(scores[field] for field in fields)])
    fields = ["split", "requested", "answered", "unsupported", "errors", "accuracy",
              "median_ms", "entropy_confidence", "uniform_chance"]
    with (REPORTS / "benchmark-source-pilot.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["source", *fields])
        for key, scores in benchmark["sources"].items():
            writer.writerow([key, *(scores.get(field) for field in fields)])
    print(f"Built {destination} ({destination.stat().st_size:,} bytes); 2 measured runs, 17 schemas.")


if __name__ == "__main__":
    main()
