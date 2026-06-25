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
)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return super().default(obj)

def safe_json(obj, **kwargs):
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def run_agent(user_query: str, filters: dict = None) -> dict:
    filters = filters or {}

    city = filters.get("city")
    max_price = filters.get("max_price")
    min_rooms = filters.get("min_rooms")
    property_type = filters.get("property_type")

    properties = search_properties(
        city=city, max_price=max_price, min_rooms=min_rooms,
        property_type=property_type, limit=15,
    )
    clusters = get_cluster_summary()
    anomalies = get_anomalies(limit=5)
    city_stats = get_city_stats()
    price_dist = get_price_distribution(city=city)

    context = f"""
אתה סוכן AI מומחה לנדל"ן ישראלי. נתח את נתוני יד2 הבאים וענה לשאלת המשתמש בעברית.

שאלת המשתמש: {user_query}

פילטרים פעילים: {safe_json(filters, ensure_ascii=False)}

נכסים תואמים (דגימה של {len(properties)}):
{safe_json(properties[:8], indent=2, ensure_ascii=False)}

סגמנטים בשוק (K-Means clustering):
{safe_json(clusters, indent=2, ensure_ascii=False)}

חריגות מחיר (Isolation Forest):
{safe_json(anomalies[:3], indent=2, ensure_ascii=False)}

סטטיסטיקות לפי עיר:
{safe_json(city_stats[:6], indent=2, ensure_ascii=False)}

התפלגות מחירים{' ב' + city if city else ''}:
{safe_json(price_dist, indent=2, ensure_ascii=False)}

הוראות:
- ענה בעברית בצורה ברורה וישירה
- התבסס על המספרים מהדאטה
- ציין האם יש עסקאות חריגות (זולות או יקרות מהממוצע)
- ציין לאיזה סגמנט שוק מתאים הצורך של המשתמש
- סיים עם המלצה ברורה "שורה תחתונה"
- כתוב 3-4 פסקאות קצרות
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
        analysis = response.json()["content"][0]["text"]
    except Exception as e:
        analysis = f"ניתוח הסוכן אינו זמין: {str(e)}"

    return {
        "analysis": analysis,
        "properties": properties[:10],
        "clusters": clusters,
        "anomalies": anomalies,
        "price_distribution": price_dist,
        "city_stats": city_stats,
        "filters_applied": filters,
    }
