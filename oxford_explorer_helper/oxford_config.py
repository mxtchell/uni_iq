"""
Oxford Economics Explorer Configuration
Metrics, dimensions, and labels for economic growth rate data
"""
import os

DATABASE_ID = os.getenv('DATABASE_ID', 'B855F1B7-35EA-46E1-B1D7-1630EEA5CA82')
TABLE_NAME = "read_csv('Oxford_Economics_Wide.csv')"

# All available metrics (growth rates)
ALL_METRICS = [
    "real_gdp",
    "cpi",
    "exchange_rate_period_average",
    "monetary_easing"
]

# Metric groups for convenience
METRIC_GROUPS = {
    "all": ALL_METRICS,
    "macro": ["real_gdp", "cpi"],
    "monetary": ["exchange_rate_period_average", "monetary_easing"]
}

# Dimensions available for breakouts
DIMENSIONS = [
    "location",
    "location_code",
    "period"
]

# Human-readable labels
METRIC_LABELS = {
    "real_gdp": "Real GDP Growth",
    "cpi": "CPI (Inflation)",
    "exchange_rate_period_average": "Exchange Rate",
    "monetary_easing": "Monetary Easing"
}

DIMENSION_LABELS = {
    "location": "Country",
    "location_code": "Country Code",
    "period": "Period"
}

# Prompt templates
INSIGHT_PROMPT_TEMPLATE = """Analyze this Oxford Economics data and provide insights:

{{facts}}

Write a brief analysis (100 words max) covering:
1. **Key Finding** - The most notable economic trend
2. **Comparison** - Notable differences across countries/periods (if breakout provided)
3. **Implication** - What this suggests for economic outlook

Use markdown formatting. Be specific with numbers."""

MAX_PROMPT_TEMPLATE = "Answer user question in 30 words or less using following facts:\n{{facts}}"
