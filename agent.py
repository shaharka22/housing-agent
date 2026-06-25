import requests
import json
import os
import numpy as np
from ml_models import (
    get_cluster_summary,
    get_anomalies,
    search_properties,
    get_city_stats,
    get_price_distribution,
    find_similar,
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)

def safe_json(obj, **kwargs):
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def run_agent(user_query: str, filters: dict = None) -> dict:
    """
    Main agent function:
    1. Gather relevant data from ML models
    2. Send to Claude with context
    3. Return structured response
    """
    filters = filters or {}

    # Gather data based on query context
    city = filters.get("city")
    max_price = filters.get("max_price")
    min_bedrooms = filters.get("min_bedrooms")
    property_type = filters.get("property_type")

    properties = search_properties(
        city=city,
        max_price=max_price,
        min_bedrooms=min_bedrooms,
        property_type=property_type,
        limit=15,
    )
    clusters = get_cluster_summary()
    anomalies = get_anomalies(limit=5)
    city_stats = get_city_stats()
    price_dist = get_price_distribution(city=city)

    # Build context for the agent
    context = f"""
You are a real estate AI analyst. Analyze the housing data below and answer the user's query with specific, data-driven insights.

USER QUERY: {user_query}

ACTIVE FILTERS: {safe_json(filters, ensure_ascii=False)}

MATCHING PROPERTIES (sample of {len(properties)}):
{safe_json(properties[:8], indent=2, ensure_ascii=False)}

MARKET CLUSTERS (K-Means segmentation):
{safe_json(clusters, indent=2, ensure_ascii=False)}

PRICE ANOMALIES (unusual deals detected by Isolation Forest):
{safe_json(anomalies[:3], indent=2, ensure_ascii=False)}

CITY-LEVEL STATS:
{safe_json(city_stats[:5], indent=2, ensure_ascii=False)}

PRICE DISTRIBUTION{' in ' + city if city else ''}:
{safe_json(price_dist, indent=2, ensure_ascii=False)}

Instructions:
- Give a clear, direct recommendation based on the data
- Reference specific numbers from the data
- Highlight any anomalies (unusually cheap/expensive) if relevant
- Mention which market segment fits the user's needs
- Keep the response to 3-4 paragraphs, professional but approachable
- End with a concrete "Bottom Line" recommendation
"""

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": context}],
    }

    try:
        response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        analysis = data["content"][0]["text"]
    except Exception as e:
        analysis = f"Agent analysis unavailable: {str(e)}. Please check your API key."

    return {
        "analysis": analysis,
        "properties": properties[:10],
        "clusters": clusters,
        "anomalies": anomalies,
        "price_distribution": price_dist,
        "city_stats": city_stats,
        "filters_applied": filters,
    }
