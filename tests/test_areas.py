"""Tests for area presets."""

import pytest
from housing_sim_jp.areas import AreaPreset, get_area, AREA_PRESETS, DEFAULT_AREA


class TestAreaPresets:
    def test_all_presets_exist(self):
        expected = {"浦和美園", "小岩", "浦和", "中野", "文京区"}
        assert set(AREA_PRESETS.keys()) == expected

    def test_default_area(self):
        assert DEFAULT_AREA == "浦和"

    def test_get_area_default(self):
        area = get_area()
        assert area.name == "浦和"

    def test_get_area_by_name(self):
        area = get_area("文京区")
        assert area.name == "文京区"
        assert area.mansion_price == 13500

    def test_get_area_unknown(self):
        with pytest.raises(ValueError, match="不明なエリア"):
            get_area("六本木")

    def test_area_is_frozen(self):
        area = get_area("浦和")
        with pytest.raises(AttributeError):
            area.mansion_price = 9999

    def test_price_ordering(self):
        """Prices should increase: 浦和美園 < 小岩 < 浦和 < 中野 < 文京区"""
        order = ["浦和美園", "小岩", "浦和", "中野", "文京区"]
        prices = [get_area(name).mansion_price for name in order]
        assert prices == sorted(prices)

    def test_mansion_instantiation_all_areas(self):
        from housing_sim_jp.strategies import Mansion
        for name in AREA_PRESETS:
            area = get_area(name)
            m = Mansion(1000, area=area)
            assert m.property_price == area.mansion_price
            assert m.name == f"{area.name}マンション"
            assert m.strategy_key == "マンション"

    def test_house_instantiation_all_areas(self):
        from housing_sim_jp.strategies import House
        for name in AREA_PRESETS:
            area = get_area(name)
            h = House(1000, area=area)
            assert h.property_price == area.house_price
            assert h.name == f"{area.name}一戸建て"
            assert h.strategy_key == "一戸建て"
