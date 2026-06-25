import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

df_cache = None
scaler = None
kmeans = None
iforest = None
feature_matrix = None

def load_and_prepare():
    global df_cache, scaler, kmeans, iforest, feature_matrix

    df = pd.read_csv("housing_data.csv")
    features = df[["price", "bedrooms", "rooms", "sqft", "med_income"]].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    cluster_labels = {
        0: "Budget Friendly",
        1: "Mid-Range Family",
        2: "Premium Urban",
        3: "Luxury Estates",
        4: "Compact Modern",
    }
    df["cluster_label"] = df["cluster"].map(cluster_labels)

    # Anomaly Detection
    iforest = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = iforest.fit_predict(X)
    df["is_anomaly"] = df["anomaly"] == -1

    # Feature matrix for similarity
    feature_matrix = X

    df_cache = df
    return df


def get_cluster_summary():
    df = df_cache
    summary = (
        df.groupby("cluster_label")
        .agg(
            count=("price", "count"),
            avg_price=("price", "mean"),
            avg_bedrooms=("bedrooms", "mean"),
            avg_sqft=("sqft", "mean"),
        )
        .round(0)
        .reset_index()
    )
    return summary.to_dict(orient="records")


def get_anomalies(limit=10):
    df = df_cache
    anomalies = df[df["is_anomaly"]].copy()
    anomalies["price_deviation"] = (
        (anomalies["price"] - df["price"].mean()) / df["price"].std()
    ).round(2)
    return anomalies.head(limit)[
        ["city", "neighborhood", "property_type", "price", "bedrooms", "sqft", "price_deviation", "description"]
    ].to_dict(orient="records")


def search_properties(city=None, max_price=None, min_bedrooms=None, property_type=None, limit=20):
    df = df_cache.copy()
    if city:
        df = df[df["city"].str.lower() == city.lower()]
    if max_price:
        df = df[df["price"] <= int(max_price)]
    if min_bedrooms:
        df = df[df["bedrooms"] >= int(min_bedrooms)]
    if property_type:
        df = df[df["property_type"].str.lower() == property_type.lower()]

    result = df.head(limit)[
        ["city", "neighborhood", "property_type", "price", "bedrooms", "sqft", "cluster_label", "is_anomaly", "description"]
    ]
    return result.to_dict(orient="records")


def find_similar(idx, top_n=5):
    if idx >= len(feature_matrix):
        return []
    target = feature_matrix[idx].reshape(1, -1)
    sims = cosine_similarity(target, feature_matrix)[0]
    sims[idx] = -1  # exclude self
    top_idx = sims.argsort()[-top_n:][::-1]
    result = df_cache.iloc[top_idx][
        ["city", "neighborhood", "property_type", "price", "bedrooms", "sqft", "cluster_label"]
    ]
    return result.to_dict(orient="records")


def get_city_stats():
    df = df_cache
    stats = (
        df.groupby("city")
        .agg(
            listings=("price", "count"),
            avg_price=("price", "mean"),
            min_price=("price", "min"),
            max_price=("price", "max"),
            anomaly_count=("is_anomaly", "sum"),
        )
        .round(0)
        .reset_index()
        .sort_values("avg_price", ascending=False)
    )
    return stats.to_dict(orient="records")


def get_price_distribution(city=None):
    df = df_cache.copy()
    if city:
        df = df[df["city"].str.lower() == city.lower()]
    bins = [0, 400000, 600000, 800000, 1000000, 1200000, 2000000]
    labels = ["<400K", "400-600K", "600-800K", "800K-1M", "1-1.2M", ">1.2M"]
    df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels)
    dist = df["price_range"].value_counts().sort_index()
    return {"labels": list(dist.index.astype(str)), "values": list(dist.values)}
