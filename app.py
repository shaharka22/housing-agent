from flask import Flask, request, jsonify, render_template_string
from flask.json.provider import DefaultJSONProvider
import ml_models
import agent as agent_module
import numpy as np
import os


class NumpyJSONProvider(DefaultJSONProvider):
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


app = Flask(__name__)
app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

# Load data on startup
print("Loading housing data and training models...")
ml_models.load_and_prepare()
print("Models ready.")

HTML = open("templates/index.html").read()


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    query = data.get("query", "What's the best city to buy a home?")
    filters = {
        k: v
        for k, v in {
            "city": data.get("city"),
            "max_price": data.get("max_price"),
            "min_rooms": data.get("min_rooms"),
            "property_type": data.get("property_type"),
        }.items()
        if v
    }
    result = agent_module.run_agent(query, filters)
    return jsonify(result)


@app.route("/api/cities")
def cities():
    stats = ml_models.get_city_stats()
    return jsonify([s["city"] for s in stats])


@app.route("/api/stats")
def stats():
    return jsonify({
        "city_stats": ml_models.get_city_stats(),
        "clusters": ml_models.get_cluster_summary(),
        "anomaly_count": len(ml_models.df_cache[ml_models.df_cache["is_anomaly"]]),
        "total_listings": len(ml_models.df_cache),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
