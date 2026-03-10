"""
Oxford Economics Explorer Functionality
Core analysis logic for economic growth rate data
"""
import pandas as pd
import jinja2
from skill_framework import SkillInput, SkillOutput, SkillVisualization, ParameterDisplayDescription
from skill_framework.layouts import wire_layout
from answer_rocket import AnswerRocketClient
from ar_analytics import ArUtils

from .oxford_config import (
    DATABASE_ID, TABLE_NAME, METRIC_GROUPS, ALL_METRICS,
    DIMENSIONS, METRIC_LABELS, DIMENSION_LABELS
)

# Brand colors
BRAND_BLUE = "#1e40af"
BRAND_SLATE = "#415A6C"


def get_label(metric):
    return METRIC_LABELS.get(metric, metric.replace('_', ' ').title())


def get_dim_label(dim):
    return DIMENSION_LABELS.get(dim, dim.replace('_', ' ').title())


def resolve_metrics(metrics_input):
    """Resolve metric input - could be a group name, single metric, or list"""
    if not metrics_input:
        return ["real_gdp"]
    if isinstance(metrics_input, str):
        if metrics_input.lower() in METRIC_GROUPS:
            return METRIC_GROUPS[metrics_input.lower()]
        return [metrics_input]
    resolved = []
    for m in metrics_input:
        if isinstance(m, str) and m.lower() in METRIC_GROUPS:
            resolved.extend(METRIC_GROUPS[m.lower()])
        else:
            resolved.append(m)
    return resolved


def clean_breakout(breakout):
    """Clean and validate breakout dimension"""
    if not breakout or str(breakout).lower() in ['none', '', 'null', 'na']:
        return None
    if breakout in DIMENSIONS:
        return breakout
    return None


def build_filter_sql(filters):
    """Build SQL filter clause from other_filters parameter"""
    if not filters:
        return "", []

    filter_conditions = []
    filter_display = []

    for f in filters:
        if isinstance(f, dict) and 'dim' in f:
            dim = f['dim']
            op = f.get('op', '=')
            values = f.get('val')

            if values is None:
                continue

            if isinstance(values, list) and values:
                if len(values) == 1:
                    filter_conditions.append(f"{dim} = '{values[0]}'")
                    filter_display.append(f"{get_dim_label(dim)}: {values[0]}")
                else:
                    values_str = "', '".join(str(v) for v in values)
                    filter_conditions.append(f"{dim} IN ('{values_str}')")
                    filter_display.append(f"{get_dim_label(dim)}: {', '.join(str(v) for v in values)}")
            elif isinstance(values, str):
                filter_conditions.append(f"{dim} = '{values}'")
                filter_display.append(f"{get_dim_label(dim)}: {values}")
            elif isinstance(values, (int, float)):
                filter_conditions.append(f"{dim} {op} {values}")
                filter_display.append(f"{get_dim_label(dim)} {op} {values}")

    if filter_conditions:
        return " AND " + " AND ".join(filter_conditions), filter_display

    return "", []


def build_param_info(metrics, breakout1, breakout2, filter_display):
    """Build parameter display descriptions for pills"""
    param_info = []

    metric_labels = [get_label(m) for m in metrics[:3]]
    if len(metrics) > 3:
        metric_labels.append(f"+{len(metrics) - 3} more")
    param_info.append(ParameterDisplayDescription(
        key="metrics",
        value=f"Metrics: {', '.join(metric_labels)}"
    ))

    breakouts = []
    if breakout1:
        breakouts.append(get_dim_label(breakout1))
    if breakout2:
        breakouts.append(get_dim_label(breakout2))
    if breakouts:
        param_info.append(ParameterDisplayDescription(
            key="breakouts",
            value=f"Breakouts: {', '.join(breakouts)}"
        ))

    if filter_display:
        param_info.append(ParameterDisplayDescription(
            key="filters",
            value=f"Filters: {'; '.join(filter_display)}"
        ))

    return param_info


def run_oxford_analysis(parameters: SkillInput) -> SkillOutput:
    """Main analysis function for Oxford Economics Explorer"""

    # Extract parameters
    metrics_input = parameters.arguments.metrics
    breakout1 = parameters.arguments.breakout_dimension
    breakout2 = getattr(parameters.arguments, 'breakout_dimension_2', None)
    filters = parameters.arguments.other_filters or []
    year = getattr(parameters.arguments, 'year', None)

    # Resolve and validate metrics
    metrics = resolve_metrics(metrics_input)
    metrics = [m for m in metrics if m in ALL_METRICS]
    if not metrics:
        metrics = ["real_gdp"]

    # Clean breakouts
    breakout1 = clean_breakout(breakout1)
    breakout2 = clean_breakout(breakout2)

    if breakout2 and not breakout1:
        breakout1, breakout2 = breakout2, None
    if breakout1 and breakout2 and breakout1 == breakout2:
        breakout2 = None

    # Determine if we need year filter (not needed if breaking out by period)
    is_time_series = breakout1 == 'period' or breakout2 == 'period'

    print(f"DEBUG: Metrics: {metrics}")
    print(f"DEBUG: Breakout1: {breakout1}, Breakout2: {breakout2}")
    print(f"DEBUG: Year: {year}, Is time series: {is_time_series}")

    # Build SQL query - these are already growth rates, just average them
    metric_selects = [f"AVG({m}) AS {m}" for m in metrics]
    group_cols = [b for b in [breakout1, breakout2] if b]

    if group_cols:
        sql_query = f"""
        SELECT {', '.join(group_cols)}, {', '.join(metric_selects)}
        FROM {TABLE_NAME} WHERE 1=1
        """
    else:
        sql_query = f"""
        SELECT {', '.join(metric_selects)}
        FROM {TABLE_NAME} WHERE 1=1
        """

    # Apply year filter if not a time series view
    year_display = None
    if not is_time_series:
        if year:
            # Filter to that year using YEAR() function
            sql_query += f" AND YEAR(period) = {year}"
            year_display = str(year)
        else:
            # Default to 2024 if no year specified
            sql_query += " AND YEAR(period) = 2024"
            year_display = "2024"

    filter_sql, filter_display = build_filter_sql(filters)
    sql_query += filter_sql

    # Add year to filter display
    if year_display:
        filter_display.insert(0, f"Year: {year_display}")

    param_info = build_param_info(metrics, breakout1, breakout2, filter_display)

    if group_cols:
        # Sort by period chronologically if it's a time series, otherwise by metric value
        if is_time_series:
            sql_query += f" GROUP BY {', '.join(group_cols)} ORDER BY period ASC"
        else:
            sql_query += f" GROUP BY {', '.join(group_cols)} ORDER BY {metrics[0]} DESC"

    print(f"DEBUG: SQL: {sql_query}")

    # Execute query
    try:
        client = AnswerRocketClient()
        result = client.data.execute_sql_query(DATABASE_ID, sql_query, row_limit=500)
        if not result.success or not hasattr(result, 'df'):
            raise Exception(f"Query failed: {getattr(result, 'error', 'Unknown')}")
        df = result.df.copy()
        print(f"DEBUG: Retrieved {len(df)} rows")
    except Exception as e:
        print(f"DEBUG: Query failed: {e}")
        return SkillOutput(final_prompt=f"Error: {e}", narrative="Error loading data.", visualizations=[])

    if len(df) == 0:
        return SkillOutput(final_prompt="No data found.", narrative="No data available.", visualizations=[])

    # Build output - growth rates can be negative, use % suffix
    suffix = "%"

    # Title
    metric_names = [get_label(m) for m in metrics]
    if len(metric_names) == 1:
        title = metric_names[0]
    elif len(metric_names) == 2:
        title = f"{metric_names[0]} & {metric_names[1]}"
    else:
        title = "Economic Indicators"

    subtitle = ""
    if breakout1:
        subtitle = f"by {get_dim_label(breakout1)}"
    if breakout2:
        subtitle += f" and {get_dim_label(breakout2)}"

    # Build chart
    chart = build_chart(df, metrics, breakout1, breakout2, suffix)

    # Build table
    columns, table_data = build_table(df, metrics, breakout1, breakout2, suffix)

    # Build narrative
    narrative_text = build_narrative(df, metrics, breakout1, breakout2, suffix)

    # Build layout
    layout = {
        "layoutJson": {
            "type": "Document",
            "style": {"padding": "20px", "fontFamily": "system-ui, -apple-system, sans-serif", "backgroundColor": "#ffffff"},
            "children": [
                {
                    "name": "HeaderContainer",
                    "type": "FlexContainer",
                    "children": "",
                    "direction": "column",
                    "style": {
                        "backgroundColor": BRAND_BLUE,
                        "padding": "20px 24px",
                        "borderRadius": "8px",
                        "marginBottom": "24px"
                    }
                },
                {
                    "name": "MainTitle",
                    "type": "Header",
                    "children": "",
                    "text": title,
                    "parentId": "HeaderContainer",
                    "style": {
                        "fontSize": "22px",
                        "fontWeight": "600",
                        "color": "#ffffff",
                        "margin": "0"
                    }
                },
                {
                    "name": "Subtitle",
                    "type": "Paragraph",
                    "children": "",
                    "text": subtitle if subtitle else "Overall Analysis",
                    "parentId": "HeaderContainer",
                    "style": {
                        "fontSize": "14px",
                        "color": "#cbd5e1",
                        "marginTop": "4px"
                    }
                },
                {
                    "name": "Chart",
                    "type": "HighchartsChart",
                    "children": "",
                    "minHeight": "400px",
                    "options": chart
                },
                {
                    "name": "TableHeader",
                    "type": "Paragraph",
                    "children": "",
                    "text": "Detailed Results",
                    "style": {
                        "fontSize": "16px",
                        "fontWeight": "600",
                        "marginTop": "28px",
                        "marginBottom": "12px",
                        "color": BRAND_SLATE
                    }
                },
                {
                    "name": "ResultsTable",
                    "type": "DataTable",
                    "children": "",
                    "columns": columns,
                    "data": table_data
                }
            ]
        },
        "inputVariables": []
    }

    try:
        html = wire_layout(layout, {})
    except Exception as e:
        html = f"<div>Error: {e}</div>"

    # Summary
    if breakout1 and breakout2:
        summary = f"Analyzed {len(metrics)} metric(s) by {get_dim_label(breakout1)} and {get_dim_label(breakout2)}."
    elif breakout1:
        summary = f"Analyzed {len(metrics)} metric(s) across {len(df)} {get_dim_label(breakout1)} segments."
    else:
        summary = f"Analyzed {len(metrics)} metric(s) across {int(df['row_count'].iloc[0]):,} data points."

    # Build facts dataframe for insights
    facts_df = build_facts_df(df, metrics, breakout1, breakout2, suffix)
    insights_dfs = [facts_df]

    facts_list = [facts_df.to_dict(orient='records')]
    insight_template = jinja2.Template(parameters.arguments.insight_prompt).render(facts=facts_list)
    max_response_prompt = jinja2.Template(parameters.arguments.max_prompt).render(facts=facts_list)

    # Generate insights using LLM
    ar_utils = ArUtils()
    generated_insights = ar_utils.get_llm_response(insight_template)

    return SkillOutput(
        final_prompt=max_response_prompt,
        narrative=generated_insights,
        visualizations=[SkillVisualization(title="Oxford Economics Explorer", layout=html)],
        insights_dfs=insights_dfs,
        parameter_display_descriptions=param_info
    )


def build_chart(df, metrics, breakout1, breakout2, suffix):
    """Build Highcharts configuration"""

    colors = [BRAND_BLUE, "#60a5fa", "#34d399", "#fbbf24", "#a78bfa", "#f87171", BRAND_SLATE]

    if not breakout1:
        # No breakout - column chart
        categories = [get_label(m) for m in metrics]
        values = [round(float(df[m].iloc[0]), 2) if pd.notna(df[m].iloc[0]) else 0 for m in metrics]
        return {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 380},
            "title": {"text": ""},
            "xAxis": {"categories": categories, "labels": {"style": {"fontSize": "12px", "color": BRAND_SLATE}}},
            "yAxis": {"title": {"text": "Growth Rate %", "style": {"color": BRAND_SLATE}}, "plotLines": [{"value": 0, "color": "#94a3b8", "width": 1}]},
            "series": [{"name": "Value", "data": values, "colorByPoint": True, "colors": colors}],
            "legend": {"enabled": False},
            "credits": {"enabled": False},
            "tooltip": {"valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,0.95)", "borderColor": BRAND_SLATE},
            "plotOptions": {"column": {"dataLabels": {"enabled": True, "format": "{y:.2f}" + suffix, "style": {"fontWeight": "500", "color": BRAND_SLATE}}}}
        }

    elif breakout1 and not breakout2:
        # Single breakout - column chart
        categories = df[breakout1].astype(str).tolist()
        series = []
        for i, m in enumerate(metrics):
            series.append({
                "name": get_label(m),
                "data": df[m].fillna(0).round(2).tolist(),
                "color": colors[i % len(colors)]
            })
        return {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 400},
            "title": {"text": ""},
            "xAxis": {"categories": categories, "title": {"text": get_dim_label(breakout1), "style": {"color": BRAND_SLATE}}, "labels": {"style": {"fontSize": "11px", "color": BRAND_SLATE}, "rotation": -45 if len(categories) > 10 else 0}},
            "yAxis": {"title": {"text": "Growth Rate %", "style": {"color": BRAND_SLATE}}, "plotLines": [{"value": 0, "color": "#94a3b8", "width": 1}]},
            "series": series,
            "legend": {"enabled": len(metrics) > 1},
            "credits": {"enabled": False},
            "tooltip": {"shared": True, "valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,0.95)", "borderColor": BRAND_SLATE},
            "plotOptions": {"column": {"dataLabels": {"enabled": len(df) <= 8, "format": "{y:.2f}" + suffix, "style": {"fontSize": "11px"}}}}
        }

    else:
        # Dual breakout - grouped column chart
        pri_vals = df[breakout1].unique().tolist()
        sec_vals = df[breakout2].unique().tolist()
        metric = metrics[0]
        series = []
        for i, sv in enumerate(sec_vals):
            data = []
            for pv in pri_vals:
                mask = (df[breakout1] == pv) & (df[breakout2] == sv)
                val = df.loc[mask, metric].iloc[0] if mask.any() else 0
                data.append(round(float(val), 2) if pd.notna(val) else 0)
            series.append({"name": str(sv), "data": data, "color": colors[i % len(colors)]})
        return {
            "chart": {"type": "column", "backgroundColor": "#ffffff", "height": 450},
            "title": {"text": get_label(metric), "style": {"fontSize": "16px", "color": BRAND_SLATE}},
            "xAxis": {"categories": [str(v) for v in pri_vals], "title": {"text": get_dim_label(breakout1), "style": {"color": BRAND_SLATE}}},
            "yAxis": {"title": {"text": "Growth Rate %", "style": {"color": BRAND_SLATE}}, "plotLines": [{"value": 0, "color": "#94a3b8", "width": 1}]},
            "series": series,
            "legend": {"enabled": True, "title": {"text": get_dim_label(breakout2), "style": {"color": BRAND_SLATE}}},
            "credits": {"enabled": False},
            "tooltip": {"valueSuffix": suffix, "backgroundColor": "rgba(255,255,255,0.95)", "borderColor": BRAND_SLATE},
            "plotOptions": {"column": {"dataLabels": {"enabled": len(pri_vals) <= 5, "format": "{y:.2f}" + suffix}}}
        }


def build_table(df, metrics, breakout1, breakout2, suffix):
    """Build table columns and data"""
    columns = []
    if breakout1:
        columns.append({"name": get_dim_label(breakout1)})
    if breakout2:
        columns.append({"name": get_dim_label(breakout2)})
    columns.extend([{"name": get_label(m)} for m in metrics])

    table_data = []
    if not breakout1:
        row = [f"{df[m].iloc[0]:.2f}{suffix}" if pd.notna(df[m].iloc[0]) else "N/A" for m in metrics]
        table_data.append(row)
    else:
        for _, r in df.iterrows():
            row = []
            if breakout1:
                row.append(str(r[breakout1]))
            if breakout2:
                row.append(str(r[breakout2]))
            row.extend([f"{r[m]:.2f}{suffix}" if pd.notna(r[m]) else "N/A" for m in metrics])
            table_data.append(row)

    return columns, table_data


def build_facts_df(df, metrics, breakout1, breakout2, suffix):
    """Build facts dataframe for LLM prompts"""
    facts = []

    if not breakout1:
        total = int(df['row_count'].iloc[0])
        facts.append({'fact_type': 'overview', 'detail': f'Total data points: {total:,}'})
        for m in metrics:
            val = df[m].iloc[0]
            if pd.notna(val):
                facts.append({
                    'fact_type': 'metric',
                    'metric': get_label(m),
                    'value': f'{val:.2f}{suffix}',
                    'data_points': total
                })
    else:
        for m in metrics[:3]:
            if m in df.columns:
                mx, mn = df[m].max(), df[m].min()
                if pd.notna(mx) and pd.notna(mn):
                    mx_seg = df.loc[df[m].idxmax(), breakout1]
                    mn_seg = df.loc[df[m].idxmin(), breakout1]
                    facts.append({
                        'fact_type': 'comparison',
                        'metric': get_label(m),
                        'highest_segment': str(mx_seg),
                        'highest_value': f'{mx:.2f}{suffix}',
                        'lowest_segment': str(mn_seg),
                        'lowest_value': f'{mn:.2f}{suffix}',
                        'gap': f'{mx - mn:.2f} points'
                    })

    return pd.DataFrame(facts)


def build_narrative(df, metrics, breakout1, breakout2, suffix):
    """Build insights narrative"""
    parts = []

    if not breakout1:
        total = int(df['row_count'].iloc[0])
        vals = [(get_label(m), float(df[m].iloc[0])) for m in metrics if pd.notna(df[m].iloc[0])]
        vals.sort(key=lambda x: x[1], reverse=True)

        parts.append(f"Analysis of **{total:,}** data points shows **{vals[0][0]}** at **{vals[0][1]:.2f}{suffix}**")
        if len(vals) > 1:
            parts.append(f", while **{vals[-1][0]}** is at **{vals[-1][1]:.2f}{suffix}**.")
        else:
            parts.append(".")

    elif breakout1 and not breakout2:
        num_segments = len(df)
        parts.append(f"Comparing **{num_segments}** {get_dim_label(breakout1)} segments:\n\n")

        for m in metrics[:2]:
            if m in df.columns:
                mx, mn = df[m].max(), df[m].min()
                if pd.notna(mx) and pd.notna(mn):
                    mx_seg = df.loc[df[m].idxmax(), breakout1]
                    mn_seg = df.loc[df[m].idxmin(), breakout1]
                    gap = mx - mn
                    parts.append(f"- **{get_label(m)}**: {mx_seg} highest at {mx:.2f}{suffix}, {mn_seg} lowest at {mn:.2f}{suffix} ({gap:.2f}pt gap)\n")

    else:
        parts.append(f"Cross-analysis by **{get_dim_label(breakout1)}** and **{get_dim_label(breakout2)}**:\n\n")
        m = metrics[0]
        if m in df.columns:
            mx, mn = df[m].max(), df[m].min()
            if pd.notna(mx) and pd.notna(mn):
                mx_r, mn_r = df.loc[df[m].idxmax()], df.loc[df[m].idxmin()]
                gap = mx - mn
                parts.append(f"**{get_label(m)}** ranges from **{mn:.2f}{suffix}** ({mn_r[breakout1]} / {mn_r[breakout2]}) to **{mx:.2f}{suffix}** ({mx_r[breakout1]} / {mx_r[breakout2]}). ")

    return "".join(parts)
