from __future__ import annotations
from skill_framework import skill, SkillParameter, SkillInput, SkillOutput

from oxford_explorer_helper.oxford_functionality import run_oxford_analysis
from oxford_explorer_helper.oxford_config import INSIGHT_PROMPT_TEMPLATE, MAX_PROMPT_TEMPLATE

@skill(
    name="Oxford Economics Explorer",
    llm_name="oxford_explorer",
    description="Analyzes Oxford Economics growth rate data including Real GDP, CPI (inflation), Exchange Rates, and Monetary Easing across countries and time periods.",
    capabilities="Compare economic indicators across countries. Analyze trends over time periods. Generate column charts for comparisons. Provide narrative insights and detailed data tables.",
    limitations="Data contains growth rates, not absolute values. Some metrics may have missing values for certain country/period combinations.",
    example_questions="What is real GDP growth by country for 2024? How does inflation compare across countries in 2023? Show GDP trends over time for the US.",
    parameter_guidance="IMPORTANT: Always specify a year parameter (e.g., 2024, 2023). Use breakout_dimension='location' to compare across countries. Use breakout_dimension='period' to see trends over time (will show all years).",
    parameters=[
        SkillParameter(
            name="year",
            description="Year to analyze (e.g., 2024, 2023, 2022). REQUIRED unless breakout_dimension is 'period'. Use most recent year (2025) if user doesn't specify.",
            is_required=False
        ),
        SkillParameter(
            name="metrics",
            description="Metrics to analyze: real_gdp, cpi, exchange_rate_period_average, monetary_easing. Can also use group names: 'all', 'macro' (gdp+cpi), 'monetary' (exchange+easing).",
            is_multi=True
        ),
        SkillParameter(
            name="breakout_dimension",
            constrained_to="dimension",
            description="Primary dimension to break out results by (location, period)"
        ),
        SkillParameter(
            name="breakout_dimension_2",
            constrained_to="dimension",
            description="Secondary dimension for cross-tabulation"
        ),
        SkillParameter(
            name="other_filters",
            constrained_to="filters",
            description="Additional filters (e.g., filter to specific countries)"
        ),
        SkillParameter(
            name="insight_prompt",
            parameter_type="prompt",
            description="Custom prompt for generating insights",
            default_value=INSIGHT_PROMPT_TEMPLATE
        ),
        SkillParameter(
            name="max_prompt",
            parameter_type="prompt",
            description="Prompt for max mode responses",
            default_value=MAX_PROMPT_TEMPLATE
        )
    ]
)
def oxford_explorer(parameters: SkillInput) -> SkillOutput:
    """Oxford Economics Explorer - analyze economic growth rate data"""
    return run_oxford_analysis(parameters)


if __name__ == '__main__':
    from skill_framework import preview_skill

    skill_input: SkillInput = oxford_explorer.create_input(arguments={
        'year': 2024,
        'metrics': ["real_gdp", "cpi"],
        'breakout_dimension': "location",
        'breakout_dimension_2': None,
        'other_filters': []
    })
    out = oxford_explorer(skill_input)
    preview_skill(oxford_explorer, out)
