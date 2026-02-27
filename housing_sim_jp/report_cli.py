"""CLI entry point for automated report generation."""

import argparse
import sys
from pathlib import Path

from housing_sim_jp.report import build_report_context, render_report


ALL_CONFIGS = [
    (Path("config.toml"), ""),
    (Path("config.example-25.toml"), "25"),
    (Path("config.example-30.toml"), "30"),
    (Path("config.example-35.toml"), "35"),
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="住宅シミュレーション レポート自動生成")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="TOML設定ファイル（--all と排他）",
    )
    parser.add_argument(
        "--name", type=str, default="",
        help="出力ファイル名のサフィックス（例: 30 → report-30.md）",
    )
    parser.add_argument(
        "--all", action="store_true", dest="run_all",
        help="4設定（config.toml, config.example-{25,30,35}.toml）を一括生成",
    )
    parser.add_argument(
        "--no-mc", action="store_true",
        help="Monte Carlo を省略（第4章スキップ、高速）",
    )
    parser.add_argument(
        "--mc-runs", type=int, default=1000,
        help="Monte Carlo 回数 (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="乱数シード (default: 42)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports"),
        help="レポート出力ディレクトリ (default: reports)",
    )
    parser.add_argument(
        "--chart-dir", type=Path, default=Path("reports/charts"),
        help="チャート出力ディレクトリ (default: reports/charts)",
    )
    return parser


def _generate_one(
    config_path: Path,
    name: str,
    *,
    no_mc: bool,
    mc_runs: int,
    seed: int,
    output_dir: Path,
    chart_dir: Path,
) -> Path:
    """Generate a single report and return the output path."""
    suffix = f"-{name}" if name else ""
    out_path = output_dir / f"report{suffix}.md"

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"レポート生成: {config_path} → {out_path}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)

    ctx = build_report_context(
        config_path=config_path,
        name=name,
        no_mc=no_mc,
        mc_runs=mc_runs,
        seed=seed,
        chart_dir=chart_dir,
    )
    md = render_report(ctx)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"  → {out_path}", file=sys.stderr)
    return out_path


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.run_all and args.config:
        parser.error("--all と --config は同時に指定できません")

    if args.run_all:
        paths = []
        for config_path, name in ALL_CONFIGS:
            if not config_path.exists():
                print(f"  スキップ: {config_path}（ファイルなし）", file=sys.stderr)
                continue
            p = _generate_one(
                config_path, name,
                no_mc=args.no_mc, mc_runs=args.mc_runs, seed=args.seed,
                output_dir=args.output, chart_dir=args.chart_dir,
            )
            paths.append(p)
        print(f"\n完了: {len(paths)}件のレポートを生成", file=sys.stderr)
        for p in paths:
            print(f"  {p}", file=sys.stderr)
    else:
        config_path = args.config or Path("config.toml")
        if not config_path.exists():
            parser.error(f"設定ファイルが見つかりません: {config_path}")
        _generate_one(
            config_path, args.name,
            no_mc=args.no_mc, mc_runs=args.mc_runs, seed=args.seed,
            output_dir=args.output, chart_dir=args.chart_dir,
        )
        print("完了", file=sys.stderr)


if __name__ == "__main__":
    main()
