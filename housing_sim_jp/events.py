"""Life event risk modeling for Monte Carlo simulation."""

from dataclasses import dataclass, field
from random import Random

from housing_sim_jp.params import SimulationParams
from housing_sim_jp.simulation import REEMPLOYMENT_AGE, PENSION_AGE


@dataclass
class EventRiskConfig:
    """Probability parameters for life event risks."""

    job_loss_annual_prob: float = 0.02
    job_loss_duration_months: int = 6
    job_loss_max_occurrences: int = 2
    disaster_annual_prob: float = 0.005
    disaster_damage_ratio: float = 0.30
    disaster_insurance_coverage: float = 0.50
    care_annual_prob_after_75: float = 0.05
    care_cost_monthly: float = 15.0
    rental_rejection_prob_after_70: float = 0.10
    rental_rejection_premium: float = 5.0
    # Divorce / spouse death
    divorce_annual_prob: float = 0.01       # 年1%（生涯≈35%）
    spouse_death_annual_prob: float = 0.003  # 年0.3%（夫婦合計）
    life_insurance_payout: float = 3000      # 生命保険金（万円）
    survivor_pension_annual: float = 75      # 遺族年金（万円/年、簡易定額）
    # Relocation
    relocation_annual_prob: float = 0.03    # 一般大企業 年3%（転勤族は10%に上昇）
    relocation_cost: float = 40.0           # 引越し費用（万円）


@dataclass
class EventTimeline:
    """Pre-sampled event timeline for a single simulation run."""

    job_loss_months: set[int] = field(default_factory=set)
    disaster_events: dict[int, float] = field(default_factory=dict)
    care_start_month: int | None = None
    rental_rejection_month: int | None = None
    care_cost_monthly: float = 15.0
    rental_rejection_premium: float = 5.0
    # Divorce / spouse death (mutually exclusive)
    divorce_month: int | None = None
    spouse_death_month: int | None = None
    life_insurance_payout: float = 3000
    survivor_pension_annual: float = 75
    # Relocation (independent of divorce/death)
    relocation_month: int | None = None
    relocation_cost: float = 40.0

    def get_extra_cost(self, month: int, age: int, params: SimulationParams) -> float:
        """Calculate extra monthly cost from care and rental rejection events."""
        cost = 0.0
        if self.care_start_month is not None and month >= self.care_start_month:
            years_from_start = month / 12
            inflation = (1 + params.inflation_rate) ** years_from_start
            cost += self.care_cost_monthly * inflation
        if self.rental_rejection_month is not None and month >= self.rental_rejection_month:
            years_from_start = month / 12
            inflation = (1 + params.inflation_rate) ** years_from_start
            cost += self.rental_rejection_premium * inflation
        return cost


def sample_events(
    rng: Random,
    config: EventRiskConfig,
    start_age: int,
    total_months: int,
    is_rental: bool,
) -> EventTimeline:
    """Sample a complete event timeline for one simulation run."""
    timeline = EventTimeline(
        care_cost_monthly=config.care_cost_monthly,
        rental_rejection_premium=config.rental_rejection_premium,
    )
    total_years = total_months // 12

    # Job loss (working age only: < REEMPLOYMENT_AGE)
    occurrences = 0
    for year_idx in range(total_years):
        age = start_age + year_idx
        if age >= REEMPLOYMENT_AGE:
            break
        if occurrences >= config.job_loss_max_occurrences:
            break
        if rng.random() < config.job_loss_annual_prob:
            start_month = year_idx * 12
            for m in range(config.job_loss_duration_months):
                if start_month + m < total_months:
                    timeline.job_loss_months.add(start_month + m)
            occurrences += 1

    # Disaster (property owners only)
    if not is_rental:
        for year_idx in range(total_years):
            if rng.random() < config.disaster_annual_prob:
                net_damage = config.disaster_damage_ratio * (
                    1 - config.disaster_insurance_coverage
                )
                timeline.disaster_events[year_idx] = net_damage

    # Care need (75+ only)
    for year_idx in range(total_years):
        age = start_age + year_idx
        if age < 75:
            continue
        if rng.random() < config.care_annual_prob_after_75:
            timeline.care_start_month = year_idx * 12
            break

    # Rental rejection premium (PENSION_AGE+ renters only)
    if is_rental:
        for year_idx in range(total_years):
            age = start_age + year_idx
            if age < PENSION_AGE:
                continue
            if rng.random() < config.rental_rejection_prob_after_70:
                timeline.rental_rejection_month = year_idx * 12
                break

    # Divorce / spouse death (mutually exclusive, stop at PENSION_AGE)
    timeline.life_insurance_payout = config.life_insurance_payout
    timeline.survivor_pension_annual = config.survivor_pension_annual
    for year_idx in range(total_years):
        age = start_age + year_idx
        if age >= PENSION_AGE:
            break
        if rng.random() < config.divorce_annual_prob:
            timeline.divorce_month = year_idx * 12
            break
        if rng.random() < config.spouse_death_annual_prob:
            timeline.spouse_death_month = year_idx * 12
            break

    # Relocation (working age only, max 1 occurrence, opt-in)
    timeline.relocation_cost = config.relocation_cost
    if config.relocation_annual_prob > 0:
        for year_idx in range(total_years):
            age = start_age + year_idx
            if age >= REEMPLOYMENT_AGE:
                break
            if rng.random() < config.relocation_annual_prob:
                timeline.relocation_month = year_idx * 12
                break

    return timeline
