"""TOML config loader with CLI > config > default resolution."""

import argparse
import tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")

DEFAULTS = {
    "age": 30,
    "savings": 800.0,
    "income": 62.5,
    "children": "32,35",
    "living": 27.0,
    "child_living": 5.0,
    "education": 10.0,
    "car": False,
    "relocation": False,
    "ideco": 4.0,
    "emergency_fund": 6.0,
}


def load_config(path: Path | None = None) -> dict:
    """Load TOML config file. Returns empty dict if file doesn't exist."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    # Normalize children: TOML list/bool/string → CLI-compatible string
    if "children" in raw:
        v = raw["children"]
        if isinstance(v, list):
            raw["children"] = ",".join(str(x) for x in v) if v else "none"
        elif v is False:
            raw["children"] = "none"
    return raw


def create_parser(description: str) -> argparse.ArgumentParser:
    """Create argparse parser with shared simulation flags."""
    d = DEFAULTS
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None, help="設定ファイルパス (default: config.toml)")
    parser.add_argument("--age", type=int, default=None, help=f"開始年齢 (default: {d['age']})")
    parser.add_argument("--savings", type=float, default=None, help=f"初期金融資産・万円 (default: {d['savings']:.0f})")
    parser.add_argument("--income", type=float, default=None, help=f"現在の世帯月額手取り・万円 (default: {d['income']})")
    parser.add_argument("--children", type=str, default=None, help=f"出産時の親の年齢（カンマ区切り、例: 28,32 / noneで子なし）(default: {d['children']})")
    parser.add_argument("--living", type=float, default=None, help=f"夫婦の生活費（万円/月、住居費・教育費・子供分除く）(default: {d['living']})")
    parser.add_argument("--child-living", type=float, default=None, help=f"子1人あたりの追加生活費（万円/月）(default: {d['child_living']})")
    parser.add_argument("--education", type=float, default=None, help=f"教育費（万円/月/人）(default: {d['education']})")
    parser.add_argument("--car", action="store_true", default=None, help="車所有（購入300万/7年買替+維持費5万/月を計上）")
    parser.add_argument("--relocation", action="store_true", default=None, help="転勤族モード（転勤確率が年3%%→10%%に上昇）")
    parser.add_argument("--ideco", type=float, default=None, help=f"iDeCo拠出額（夫婦合計・万円/月）(default: {d['ideco']})")
    parser.add_argument("--emergency-fund", type=float, default=None, help=f"生活防衛資金（生活費の何ヶ月分）(default: {d['emergency_fund']})")
    return parser


def parse_args(description: str) -> tuple[dict, list[int]]:
    """Parse CLI args, load config, resolve values, and return (resolved_dict, child_birth_ages)."""
    parser = create_parser(description)
    args = parser.parse_args()
    config = load_config(args.config)
    r = resolve(args, config)
    children_str = str(r["children"]).strip().lower()
    child_birth_ages = [] if children_str == "none" else [int(x) for x in children_str.split(",")]
    return r, child_birth_ages


def resolve(args: argparse.Namespace, config: dict) -> dict:
    """Resolve values with priority: CLI flag > config.toml > hardcoded default."""
    resolved = {}
    for key, default in DEFAULTS.items():
        cli_val = getattr(args, key, None)
        resolved[key] = cli_val if cli_val is not None else config.get(key, default)
    return resolved
