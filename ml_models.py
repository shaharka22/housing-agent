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

    # Clean data
    df = df.dropna(subset=["price", "rooms", "area_sqm", "price_per_sqrm"])
    df = df[df["price"] > 0]
    df = df[df["area_sqm"] < 10000]  # remove extreme outliers
    df = df[df["price"] < 50000000]  # remove extreme outliers

    # Fill missing floor with median
    df["floor"] = df["floor"].fillna(df["floor"].median())

    features = df[["price", "rooms", "area_sqm", "price_per_sqrm", "floor"]].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    # Label clusters by avg price
    cluster_avg = df.groupby("cluster")["price"].mean().sort_values()
    cluster_labels = {}
    labels = ["חסכוני", "בינוני", "מעל הממוצע", "יוקרה", "פרימיום"]
    for i, (cluster_id, _) in enumerate(cluster_avg.items()):
        cluster_labels[cluster_id] = labels[i]
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
            avg_rooms=("rooms", "mean"),
            avg_sqm=("area_sqm", "mean"),
        )
        .round(0)
        .reset_index()
        .sort_values("avg_price")
    )
    return summary.to_dict(orient="records")


def get_anomalies(limit=10):
    df = df_cache
    anomalies = df[df["is_anomaly"]].copy()
    anomalies["price_deviation"] = (
        (anomalies["price"] - df["price"].mean()) / df["price"].std()
    ).round(2)
    return anomalies.head(limit)[
        ["city", "neighborhood", "property_type", "price", "rooms", "area_sqm", "price_per_sqrm", "price_deviation"]
    ].to_dict(orient="records")


def search_properties(city=None, max_price=None, min_rooms=None, property_type=None, limit=20):
    df = df_cache.copy()
    if city:
        df = df[df["city"].str.contains(city, na=False)]
    if max_price:
        df = df[df["price"] <= int(max_price)]
    if min_rooms:
        df = df[df["rooms"] >= float(min_rooms)]
    if property_type:
        df = df[df["property_type"].str.contains(property_type, na=False)]

    result = df.head(limit)[
        ["city", "neighborhood", "property_type", "price", "rooms", "area_sqm",
         "price_per_sqrm", "floor", "cluster_label", "is_anomaly"]
    ]
    return result.to_dict(orient="records")


def find_similar(idx, top_n=5):
    if idx >= len(feature_matrix):
        return []
    target = feature_matrix[idx].reshape(1, -1)
    sims = cosine_similarity(target, feature_matrix)[0]
    sims[idx] = -1
    top_idx = sims.argsort()[-top_n:][::-1]
    result = df_cache.iloc[top_idx][
        ["city", "neighborhood", "property_type", "price", "rooms", "area_sqm", "cluster_label"]
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
            avg_price_per_sqm=("price_per_sqrm", "mean"),
            anomaly_count=("is_anomaly", "sum"),
        )
        .round(0)
        .reset_index()
        .sort_values("listings", ascending=False)
        .head(15)
    )
    return stats.to_dict(orient="records")


def get_price_distribution(city=None):
    df = df_cache.copy()
    if city:
        df = df[df["city"].str.contains(city, na=False)]
    bins = [0, 1000000, 2000000, 3000000, 4000000, 6000000, 100000000]
    labels = ["עד 1M", "1-2M", "2-3M", "3-4M", "4-6M", "מעל 6M"]
    df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels)
    dist = df["price_range"].value_counts().sort_index()
    return {"labels": list(dist.index.astype(str)), "values": list(dist.values)}
