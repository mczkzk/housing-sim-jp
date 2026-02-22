"""TOML config loader with CLI > config > default resolution."""

import argparse
import tomllib
from pathlib import Path

DEFAULT_CONFIG_PATH = Path("config.toml")

DEFAULTS = {
    "husband_age": 30,
    "wife_age": 28,
    "savings": 800.0,
    "husband_income": 40.0,
    "wife_income": 22.5,
    "children": "32,35",
    "living_premium": 0.0,
    "child_living": 5.0,
    "education": 10.0,
    "car": False,
    "pets": 0,
    "relocation": False,
    "husband_ideco": 2.0,
    "wife_ideco": 2.0,
    "emergency_fund": 6.0,
    "special_expenses": "",
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
    # Normalize special_expenses: TOML [[age, amount, label?], ...] → "age:amount:label,..." string
    if "special_expenses" in raw:
        v = raw["special_expenses"]
        if isinstance(v, list):
            parts = []
            for pair in v:
                age, amount = int(pair[0]), pair[1]
                label = pair[2] if len(pair) >= 3 else ""
                parts.append(f"{age}:{amount}:{label}" if label else f"{age}:{amount}")
            raw["special_expenses"] = ",".join(parts) if parts else ""
    return raw


def create_parser(description: str) -> argparse.ArgumentParser:
    """Create argparse parser with shared simulation flags."""
    d = DEFAULTS
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, default=None, help="設定ファイルパス (default: config.toml)")
    parser.add_argument("--husband-age", type=int, default=None, help=f"夫の開始年齢 (default: {d['husband_age']})")
    parser.add_argument("--wife-age", type=int, default=None, help=f"妻の開始年齢 (default: {d['wife_age']})")
    parser.add_argument("--savings", type=float, default=None, help=f"初期金融資産・万円 (default: {d['savings']:.0f})")
    parser.add_argument("--husband-income", type=float, default=None, help=f"夫の月額手取り・万円 (default: {d['husband_income']})")
    parser.add_argument("--wife-income", type=float, default=None, help=f"妻の月額手取り・万円 (default: {d['wife_income']})")
    parser.add_argument("--children", type=str, default=None, help=f"出産時の親の年齢（カンマ区切り、例: 28,32 / noneで子なし）(default: {d['children']})")
    parser.add_argument("--living-premium", type=float, default=None, help=f"生活費プレミアム（年齢別ベースラインへの上乗せ、万円/月）(default: {d['living_premium']})")
    parser.add_argument("--child-living", type=float, default=None, help=f"子1人あたりの追加生活費（万円/月）(default: {d['child_living']})")
    parser.add_argument("--education", type=float, default=None, help=f"教育費（万円/月/人）(default: {d['education']})")
    parser.add_argument("--car", action="store_true", default=None, help="車所有（購入300万/7年買替+維持費5万/月を計上）")
    parser.add_argument("--pets", type=int, default=None, help=f"ペット頭数（1匹15年・飼育費1.5万/月、賃貸は+1.5万/月）(default: {d['pets']})")
    parser.add_argument("--relocation", action="store_true", default=None, help="転勤族モード（転勤確率が年3%%→10%%に上昇）")
    parser.add_argument("--husband-ideco", type=float, default=None, help=f"夫のiDeCo拠出額（万円/月）(default: {d['husband_ideco']})")
    parser.add_argument("--wife-ideco", type=float, default=None, help=f"妻のiDeCo拠出額（万円/月）(default: {d['wife_ideco']})")
    parser.add_argument("--emergency-fund", type=float, default=None, help=f"生活防衛資金（生活費の何ヶ月分）(default: {d['emergency_fund']})")
    parser.add_argument("--special-expenses", type=str, default=None, help="特別支出（年齢:金額のカンマ区切り、例: 55:500,65:300）")
    return parser


def parse_special_expenses(s: str) -> dict[int, float]:
    """Parse special expenses string "age:amount[:label],..." → {age: amount}."""
    if not s or not s.strip():
        return {}
    result: dict[int, float] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(":")
        age = int(parts[0].strip())
        amount = float(parts[1].strip())
        result[age] = result.get(age, 0) + amount
    return result


def parse_special_expense_labels(s: str) -> list[tuple[int, float, str]]:
    """Parse special expenses string → [(age, amount, label), ...] for chart annotations."""
    if not s or not s.strip():
        return []
    result: list[tuple[int, float, str]] = []
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        parts = pair.split(":")
        age = int(parts[0].strip())
        amount = float(parts[1].strip())
        label = parts[2].strip() if len(parts) >= 3 else f"{amount:.0f}万"
        result.append((age, amount, label))
    return sorted(result)


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
