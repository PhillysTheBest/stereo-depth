"""Run an engine over a dataset and report EPE, D1-all, bad-2.0 and runtime.

uv run stereo-bench --dataset kitti2015 --root data/kitti2015 --engine sgbm raft_stereo

Several engines print a comparison table after the per-pair scores.
"""

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from stereo_depth.config import Settings
from stereo_depth.core.resize import downscale, upscale_disparity
from stereo_depth.engines import DepthEngine
from stereo_depth.engines.registry import get_engines
from stereo_depth.eval.datasets import DATASETS, StereoSample
from stereo_depth.eval.metrics import PairScores, mean_scores, score


def evaluate(
    engine: DepthEngine, samples: Iterable[StereoSample], max_px: int, limit: int | None = None
) -> list[tuple[str, PairScores]]:
    results: list[tuple[str, PairScores]] = []
    for i, sample in enumerate(samples):
        if limit is not None and i >= limit:
            break
        results.append((sample.name, run_pair(engine, sample, max_px)))
    return results


def run_pair(engine: DepthEngine, sample: StereoSample, max_px: int) -> PairScores:
    left, scale = downscale(sample.left, max_px)
    right, _ = downscale(sample.right, max_px)
    out = engine.predict(left, right)
    disp = upscale_disparity(out.values, sample.gt.shape, scale)
    return score(disp, sample.gt, sample.valid, out.runtime_ms)


def _log_wandb(project: str, config: dict[str, Any], summary: dict[str, float]) -> None:
    try:
        import wandb
    except ImportError:
        print("wandb not installed; install with `uv sync --extra track`", file=sys.stderr)
        return
    run = wandb.init(project=project, config=config, job_type="bench")
    run.summary.update(summary)
    run.finish()


def _print_pairs(
    engine: str, results: list[tuple[str, PairScores]], summary: dict[str, float]
) -> None:
    print(f"\n{engine}")
    print(f"{'pair':<16} {'EPE':>7} {'D1-all':>7} {'bad-2':>7} {'ms':>8}")
    for name, s in results:
        print(f"{name:<16} {s.epe:7.3f} {s.d1_all:7.2%} {s.bad_2:7.2%} {s.runtime_ms:8.1f}")
    print(
        f"{'mean':<16} {summary['epe']:7.3f} {summary['d1_all']:7.2%} "
        f"{summary['bad_2']:7.2%} {summary['runtime_ms']:8.1f}  ({len(results)} pairs)"
    )


def _print_comparison(dataset: str, summaries: dict[str, dict[str, float]]) -> None:
    """Markdown table, so it can be pasted into the README or a PR."""
    print(f"\n{dataset}\n")
    print("| engine | EPE | D1-all | bad-2.0 | ms/pair |")
    print("|---|---:|---:|---:|---:|")
    for name, s in summaries.items():
        print(
            f"| {name} | {s['epe']:.3f} | {s['d1_all']:.2%} | {s['bad_2']:.2%} "
            f"| {s['runtime_ms']:.0f} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--engine", nargs="+", choices=sorted(get_engines()), default=["sgbm"])
    parser.add_argument("--limit", type=int, default=None, help="score only the first N pairs")
    parser.add_argument("--max-px", type=int, default=Settings().max_width_px)
    parser.add_argument("--out", type=Path, default=None, help="write per-pair JSON here")
    parser.add_argument("--wandb", metavar="PROJECT", default=None, help="log summary to W&B")
    args = parser.parse_args(argv)

    summaries: dict[str, dict[str, float]] = {}
    for name in args.engine:
        engine = get_engines()[name]
        if not engine.is_available():
            print(f"skipping {name}: not available (missing weights?)", file=sys.stderr)
            continue
        samples = DATASETS[args.dataset](args.root)
        results = evaluate(engine, samples, args.max_px, args.limit)
        summary = mean_scores([s for _, s in results])
        summaries[name] = summary
        _print_pairs(name, results, summary)

        config = {"dataset": args.dataset, "engine": name, "max_px": args.max_px}
        if args.out:
            payload = {
                "config": config,
                "summary": summary,
                "pairs": [{"name": n, **asdict(s)} for n, s in results],
            }
            out = (
                args.out if len(args.engine) == 1 else args.out.with_stem(f"{args.out.stem}-{name}")
            )
            out.write_text(json.dumps(payload, indent=2))
        if args.wandb:
            _log_wandb(args.wandb, config, summary)

    if len(summaries) > 1:
        _print_comparison(args.dataset, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
