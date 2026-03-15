"""Area presets for housing simulation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AreaPreset:
    """Immutable area-specific parameters for housing strategies."""

    name: str
    description: str

    # マンション
    mansion_price: float
    mansion_initial_cost: float
    mansion_building_age: int
    mansion_management_fee: float
    mansion_repair_reserve: float
    mansion_property_tax: float
    mansion_insurance: float = 0.15
    mansion_land_ratio: float = 0.25
    mansion_one_time_expenses: dict[int, float] | None = None
    mansion_liquidation_cost: float = 200

    # 一戸建て
    house_price: float = 0
    house_initial_cost: float = 0
    house_building_age: int = 7
    house_property_tax: float = 1.8
    house_insurance: float = 0.4
    house_maintenance_base: float = 1.5
    house_other_monthly: float = 0.7
    house_land_ratio: float = 0.55
    house_one_time_expenses: dict[int, float] | None = None
    house_liquidation_cost: float = 650
    house_utility_premium: float = 0.3
    house_liquidity_discount: float = 0.15

    # 賃貸
    rent_2ldk: float = 18.0
    rent_3ldk: float = 25.0
    rent_extra_child: float = 2.0
    rental_initial_cost: float = 105

    def __post_init__(self):
        # frozen dataclass cannot assign, use object.__setattr__
        if self.mansion_one_time_expenses is None:
            object.__setattr__(self, 'mansion_one_time_expenses', {
                20: 40, 30: 100, 40: 80, 48: 370, 55: 100, 62: 150,
            })
        if self.house_one_time_expenses is None:
            object.__setattr__(self, 'house_one_time_expenses', {
                17: 180, 30: 500, 45: 300, 55: 400,
            })


DEFAULT_AREA = "浦和"

AREA_PRESETS: dict[str, AreaPreset] = {
    "浦和美園": AreaPreset(
        name="浦和美園",
        description="さいたま市緑区浦和美園エリア（埼玉高速鉄道沿線、郊外型）",
        mansion_price=4750,
        mansion_initial_cost=380,
        mansion_building_age=10,
        mansion_management_fee=1.2,
        mansion_repair_reserve=0.9,
        mansion_property_tax=1.2,
        mansion_land_ratio=0.15,
        house_price=4500,
        house_initial_cost=360,
        house_land_ratio=0.50,
        house_property_tax=1.2,
        rent_2ldk=12.5,
        rent_3ldk=17.0,
    ),
    "小岩": AreaPreset(
        name="小岩",
        description="東京都江戸川区小岩エリア（JR総武線、下町・再開発エリア）",
        mansion_price=7000,
        mansion_initial_cost=560,
        mansion_building_age=10,
        mansion_management_fee=1.4,
        mansion_repair_reserve=1.0,
        mansion_property_tax=1.6,
        mansion_land_ratio=0.20,
        house_price=6215,
        house_initial_cost=497,
        house_land_ratio=0.55,
        house_property_tax=1.6,
        rent_2ldk=16.0,
        rent_3ldk=21.5,
    ),
    "浦和": AreaPreset(
        name="浦和",
        description="さいたま市浦和区（JR京浜東北線・湘南新宿ライン、文教エリア）",
        mansion_price=8500,
        mansion_initial_cost=680,
        mansion_building_age=10,
        mansion_management_fee=1.55,
        mansion_repair_reserve=1.1,
        mansion_property_tax=1.8,
        mansion_land_ratio=0.25,
        house_price=7372,
        house_initial_cost=590,
        house_land_ratio=0.55,
        house_property_tax=1.8,
        rent_2ldk=18.0,
        rent_3ldk=25.0,
    ),
    "中野": AreaPreset(
        name="中野",
        description="東京都中野区（JR中央線・東西線、サブカル・利便性エリア）",
        mansion_price=10500,
        mansion_initial_cost=840,
        mansion_building_age=10,
        mansion_management_fee=1.8,
        mansion_repair_reserve=1.3,
        mansion_property_tax=2.2,
        mansion_land_ratio=0.30,
        house_price=9250,
        house_initial_cost=740,
        house_land_ratio=0.65,
        house_property_tax=2.2,
        rent_2ldk=23.0,
        rent_3ldk=31.5,
    ),
    "文京区": AreaPreset(
        name="文京区",
        description="東京都文京区（東京メトロ丸ノ内線・南北線、最高峰の文教エリア）",
        mansion_price=13500,
        mansion_initial_cost=1080,
        mansion_building_age=10,
        mansion_management_fee=2.1,
        mansion_repair_reserve=1.5,
        mansion_property_tax=2.8,
        mansion_land_ratio=0.40,
        house_price=10941,
        house_initial_cost=875,
        house_land_ratio=0.75,
        house_property_tax=2.8,
        rent_2ldk=28.5,
        rent_3ldk=40.0,
    ),
}


def get_area(name: str | None = None) -> AreaPreset:
    """Get area preset by name. Raises ValueError for unknown areas."""
    if name is None:
        name = DEFAULT_AREA
    if name not in AREA_PRESETS:
        available = ", ".join(AREA_PRESETS.keys())
        raise ValueError(f"不明なエリア: {name!r}（選択肢: {available}）")
    return AREA_PRESETS[name]
