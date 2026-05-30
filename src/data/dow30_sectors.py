"""Static sector map for the current 29-ticker Dow-style universe."""

from __future__ import annotations


DOW30_STATIC_SECTOR_MAP: dict[str, str] = {
    "AAPL": "technology",
    "AMGN": "healthcare",
    "AMZN": "consumer_discretionary",
    "AXP": "financials",
    "BA": "industrials",
    "CAT": "industrials",
    "CRM": "technology",
    "CSCO": "technology",
    "CVX": "energy",
    "DIS": "communication_services",
    "GS": "financials",
    "HD": "consumer_discretionary",
    "HON": "industrials",
    "IBM": "technology",
    "INTC": "technology",
    "JNJ": "healthcare",
    "JPM": "financials",
    "KO": "consumer_staples",
    "MCD": "consumer_discretionary",
    "MMM": "industrials",
    "MRK": "healthcare",
    "MSFT": "technology",
    "NKE": "consumer_discretionary",
    "PG": "consumer_staples",
    "TRV": "financials",
    "UNH": "healthcare",
    "V": "financials",
    "VZ": "communication_services",
    "WMT": "consumer_staples",
}


def get_sector_map(name: str) -> dict[str, str]:
    """Return a ticker -> sector map by name."""
    if name != "dow30_static":
        raise ValueError(f"Unknown sector map: {name}")
    return dict(DOW30_STATIC_SECTOR_MAP)
