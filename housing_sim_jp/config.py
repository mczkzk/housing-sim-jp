"""TOML config loader with CLI > config > default resolution."""

import argparse
import sys
import tomllib
from pathlib import Path

from housing_sim_jp.params import SimulationParams
from housing_sim_jp.simulation import GRAD_SCHOOL_MAP, DEFAULT_INDEPENDENCE_AGE

DEFAULT_CONFIG_PATH = Path("config.toml")

DEFAULTS = {
    "husband_age": 30,
    "wife_age": 28,
    "savings": 800.0,
    "husband_income": 40.0,
    "wife_income": 22.5,
    "children": "30,33",
    "living_premium": 0.0,
    "child_living": 5.0,
    "education_private_from": "",
    "education_field": "理系",
    "education_boost": 1.0,
    "education_grad": "学部",
    "car": False,
    "pets": "",
    "relocation": False,
    "husband_ideco": 2.0,
    "wife_ideco": 2.0,
    "emergency_fund": 6.0,
    "husband_pension_start_age": 60,
    "wife_pension_start_age": 60,
    "husband_work_end_age": 70,
    "wife_work_end_age": 70,
    "special_expenses": "",
}


def load_config(path: Path | None = None) -> dict:
    """Load TOML config file. Returns empty dict if file doesn't exist."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"設定ファイルの読み込みに失敗: {path}: {e}", file=sys.stderr)
        raise SystemExit(1)
    # Normalize children: TOML list/bool/string → CLI-compatible string
    # Supports: [30, 33], ["30:修士", "33:博士"], [[30, "修士"], [33]]
    if "children" in raw:
        v = raw["children"]
        if isinstance(v, list):
            parts = []
            for item in v:
                if isinstance(item, list):
                    parts.append(":".join(str(x) for x in item))
                else:
                    parts.append(str(item))
            raw["children"] = ",".join(parts) if parts else "none"
        elif v is False:
            raw["children"] = "none"
    # Normalize pets: TOML list/int/bool → CLI-compatible string
    if "pets" in raw:
        v = raw["pets"]
        if isinstance(v, list):
            raw["pets"] = ",".join(str(x) for x in v) if v else ""
        elif isinstance(v, bool) and v is False:
            raw["pets"] = ""
        elif isinstance(v, int):
            # Backward compat: bare integer → empty (0) or error guidance
            raw["pets"] = "" if v == 0 else str(v)
    # Migrate legacy education key → new 4-parameter model
    if "education" in raw and "education_private_from" not in raw:
        edu = raw.pop("education")
        if isinstance(edu, (int, float)):
            if edu <= 12:
                raw["education_private_from"] = ""
            elif edu <= 17:
                raw["education_private_from"] = "高校"
            else:
                raw["education_private_from"] = "中学"
            raw.setdefault("education_field", "理系")
            raw.setdefault("education_boost", 1.0)
    elif "education" in raw and "education_private_from" in raw:
        raw.pop("education")  # new params take precedence
    # Migrate legacy pension_start_age / work_end_age → husband_*/wife_*
    if "pension_start_age" in raw and "husband_pension_start_age" not in raw:
        v = raw.pop("pension_start_age")
        raw.setdefault("husband_pension_start_age", v)
        raw.setdefault("wife_pension_start_age", v)
    elif "pension_start_age" in raw:
        raw.pop("pension_start_age")
    if "work_end_age" in raw and "husband_work_end_age" not in raw:
        v = raw.pop("work_end_age")
        raw.setdefault("husband_work_end_age", v)
        raw.setdefault("wife_work_end_age", v)
    elif "work_end_age" in raw:
        raw.pop("work_end_age")
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
    parser.add_argument("--children", type=str, default=None, help=f"出産時の妻の年齢（カンマ区切り、例: 30,33 / noneで子なし）(default: {d['children']})")
    parser.add_argument("--living-premium", type=float, default=None, help=f"生活費プレミアム（年齢別ベースラインへの上乗せ、万円/月）(default: {d['living_premium']})")
    parser.add_argument("--child-living", type=float, default=None, help=f"子1人あたりの追加生活費（万円/月）(default: {d['child_living']})")
    parser.add_argument("--education-private-from", type=str, default=None, help="私立切替ステージ: \"\"=全公立, 中学, 高校, 大学 (default: 全公立)")
    parser.add_argument("--education-field", type=str, default=None, help="進路: 理系, 文系 (default: 理系)")
    parser.add_argument("--education-boost", type=float, default=None, help="受験年費用倍率 0.8=節約, 1.0=標準, 1.2=積極 (default: 1.0)")
    parser.add_argument("--education-grad", type=str, default=None, help="最終学歴: 学部(22歳独立), 修士(24歳), 博士(27歳) (default: 学部)")
    parser.add_argument("--car", action="store_true", default=None, help="車所有（購入300万/7年買替+維持費5万/月を計上）")
    parser.add_argument("--pets", type=str, default=None, help="ペット迎え入れ時の夫の年齢（カンマ区切り、例: 38,40 / noneでペットなし）")
    parser.add_argument("--relocation", action="store_true", default=None, help="転勤族モード（転勤確率が年3%%→10%%に上昇）")
    parser.add_argument("--husband-ideco", type=float, default=None, help=f"夫のiDeCo拠出額（万円/月）(default: {d['husband_ideco']})")
    parser.add_argument("--wife-ideco", type=float, default=None, help=f"妻のiDeCo拠出額（万円/月）(default: {d['wife_ideco']})")
    parser.add_argument("--emergency-fund", type=float, default=None, help=f"生活防衛資金（生活費の何ヶ月分）(default: {d['emergency_fund']})")
    parser.add_argument("--husband-pension-start-age", type=int, default=None, help=f"夫の年金受給開始年齢（60-75, default: {d['husband_pension_start_age']}）")
    parser.add_argument("--wife-pension-start-age", type=int, default=None, help=f"妻の年金受給開始年齢（60-75, default: {d['wife_pension_start_age']}）")
    parser.add_argument("--husband-work-end-age", type=int, default=None, help=f"夫の再雇用終了年齢（60-75, default: {d['husband_work_end_age']}）")
    parser.add_argument("--wife-work-end-age", type=int, default=None, help=f"妻の再雇用終了年齢（60-75, default: {d['wife_work_end_age']}）")
    parser.add_argument("--special-expenses", type=str, default=None, help="特別支出（年齢:金額[:ラベル]のカンマ区切り、例: 55:500:リフォーム,65:300）")
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


def parse_pet_ages(s: str) -> list[int]:
    """Parse pets string → list of husband's ages at adoption. Empty/none → []."""
    s = str(s).strip().lower()
    if not s or s == "none":
        return []
    return sorted(int(x) for x in s.split(","))



def parse_children_config(s: str) -> tuple[list[int], list[int]]:
    """Parse children string → (birth_ages, independence_ages).

    Format: "30,33:博士" → ([30, 33], [22, 27])
    Supports: plain ages, age:修士, age:博士
    """
    s = str(s).strip().lower()
    if not s or s == "none":
        return [], []
    birth_ages = []
    independence_ages = []
    for part in s.split(","):
        part = part.strip()
        if ":" in part:
            age_str, grad = part.split(":", 1)
            birth_ages.append(int(age_str))
            independence_ages.append(GRAD_SCHOOL_MAP[grad])
        else:
            birth_ages.append(int(part))
            independence_ages.append(DEFAULT_INDEPENDENCE_AGE)
    return birth_ages, independence_ages


def build_params(r: dict, pet_sim_ages: tuple[int, ...] = ()) -> SimulationParams:
    """Build SimulationParams from resolved config dict."""
    return SimulationParams(
        husband_income=r["husband_income"],
        wife_income=r["wife_income"],
        husband_pension_start_age=r["husband_pension_start_age"],
        wife_pension_start_age=r["wife_pension_start_age"],
        husband_work_end_age=r["husband_work_end_age"],
        wife_work_end_age=r["wife_work_end_age"],
        living_premium=r["living_premium"],
        child_living_cost_monthly=r["child_living"],
        education_private_from=r["education_private_from"],
        education_field=r["education_field"],
        education_boost=r["education_boost"],
        education_grad=r["education_grad"],
        has_car=r["car"],
        pet_adoption_ages=pet_sim_ages,
        husband_ideco=r["husband_ideco"],
        wife_ideco=r["wife_ideco"],
        emergency_fund_months=r["emergency_fund"],
        special_expenses=parse_special_expenses(r["special_expenses"]),
    )


def parse_args(
    description: str,
    add_args_fn: "Callable[[argparse.ArgumentParser], None] | None" = None,
) -> tuple[dict, list[int], list[int], list[int], argparse.Namespace]:
    """Parse CLI args, load config, resolve values.

    Returns (resolved_dict, child_birth_ages, independence_ages, pet_ages, namespace).
    child_birth_ages: list of wife's ages at birth.
    independence_ages: per-child independence age (22=学部, 24=修士, 27=博士).
    pet_ages: list of husband's ages at pet adoption.
    namespace: raw argparse.Namespace (for extra CLI args added via add_args_fn).
    """
    parser = create_parser(description)
    if add_args_fn:
        add_args_fn(parser)
    args = parser.parse_args()
    config = load_config(args.config)
    r = resolve(args, config)
    child_birth_ages, legacy_indep = parse_children_config(r["children"])
    # education_grad takes precedence; fall back to per-child legacy spec
    grad = r["education_grad"]
    grad_age = GRAD_SCHOOL_MAP.get(grad, DEFAULT_INDEPENDENCE_AGE)
    if grad != DEFAULTS["education_grad"] or all(a == DEFAULT_INDEPENDENCE_AGE for a in legacy_indep):
        independence_ages = [grad_age] * len(child_birth_ages)
    else:
        independence_ages = legacy_indep
    pet_ages = parse_pet_ages(r["pets"])
    return r, child_birth_ages, independence_ages, pet_ages, args


def resolve_sim_ages(
    r: dict, child_birth_ages: list[int], pet_ages: list[int],
) -> tuple[int, list[int], tuple[int, ...]]:
    """Derive start_age and convert child/pet ages to sim-age basis.

    Returns (start_age, child_sim_ages, pet_sim_ages).
    """
    from housing_sim_jp.simulation import to_sim_ages

    husband_age = r["husband_age"]
    wife_age = r["wife_age"]
    start_age = max(husband_age, wife_age)
    child_sim_ages = to_sim_ages(child_birth_ages, wife_age, start_age)
    pet_sim_ages = tuple(sorted(to_sim_ages(pet_ages, husband_age, start_age)))
    return start_age, child_sim_ages, pet_sim_ages


def resolve(args: argparse.Namespace, config: dict) -> dict:
    """Resolve values with priority: CLI flag > config.toml > hardcoded default."""
    resolved = {}
    for key, default in DEFAULTS.items():
        cli_val = getattr(args, key, None)
        resolved[key] = cli_val if cli_val is not None else config.get(key, default)
    return resolved
