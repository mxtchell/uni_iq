"""
Oxford Economics Explorer Configuration
Metrics, dimensions, and labels for economic growth rate data
"""
import os

DATABASE_ID = os.getenv('DATABASE_ID', '3D9EB9FB-9E96-46FC-8F37-E65422403C73')
TABLE_NAME = "`pds_aurora_931914_dev`.`finance`.`oxford_economics_wide`"

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

# Metrics that are percentages (get % suffix)
PERCENT_METRICS = ["real_gdp", "cpi", "monetary_easing"]

# Metrics that are raw numbers (no % suffix)
NUMBER_METRICS = ["exchange_rate_period_average"]

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
