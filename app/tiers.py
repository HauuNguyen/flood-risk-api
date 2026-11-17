"""Tier definitions shared by the offline pipeline and the API (no heavy dependencies)."""
TIERS = ["Low (bottom 50%)", "Moderate", "High (top 20%)", "Very High (top 5%)"]
TIER_BINS = [0, 0.50, 0.80, 0.95, 1.0]
DISCLAIMER = ("Ranking of long-run flood-proneness for the target year. It is NOT a forecast of how many Tambons "
              "will flood or of a specific flood event: the national year-to-year flood level is not predictable "
              "from history.")


def tier_from_pct(pct: float) -> str:
    for label, upper in zip(TIERS, TIER_BINS[1:]):
        if pct <= upper:
            return label
    return TIERS[-1]
