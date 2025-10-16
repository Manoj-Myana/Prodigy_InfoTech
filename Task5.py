# Install needed libraries
!pip install kagglehub geopandas folium shapely seaborn --quiet

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap
import geopandas as gpd
from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import chi2_contingency
from sklearn.cluster import DBSCAN
import kagglehub

# Download dataset
path = kagglehub.dataset_download("sobhanmoosavi/us-accidents")
csv_path = os.path.join(path, "US_Accidents_March23.csv")

# Load a sample of data to avoid memory issues
acc = pd.read_csv(csv_path, nrows=200_000)
# Force datetime conversion with coercion
acc["Start_Time"] = pd.to_datetime(acc["Start_Time"], errors="coerce")
acc["End_Time"] = pd.to_datetime(acc["End_Time"], errors="coerce")

# Feature engineering
acc["hour"] = acc["Start_Time"].dt.hour
acc["dayofweek"] = acc["Start_Time"].dt.dayofweek
acc["date"] = acc["Start_Time"].dt.date

def simplify_weather(w):
    if pd.isna(w):
        return "Unknown"
    w = w.lower()
    if "rain" in w or "drizzle" in w:
        return "Rain"
    elif "snow" in w:
        return "Snow"
    elif "fog" in w or "mist" in w:
        return "Fog"
    elif "clear" in w or "sun" in w or "fair" in w:
        return "Clear"
    elif "cloud" in w:
        return "Cloudy"
    else:
        return "Other"

acc["weather_simple"] = acc["Weather_Condition"].apply(simplify_weather)

# Safely create road_surface column
if "Road_Surface_Conditions" in acc.columns:
    acc["road_surface"] = acc["Road_Surface_Conditions"].fillna("Unknown")
else:
    acc["road_surface"] = "Unknown"

# Set seaborn style
sns.set(style="whitegrid")

# === Visualizations & Analyses ===

# 1. Accident count by hour
plt.figure(figsize=(10, 4))
sns.countplot(x="hour", data=acc, order=range(24))
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Count")
plt.show()

# 2. Accident count by day of week
plt.figure(figsize=(8, 4))
sns.countplot(x="dayofweek", data=acc)
plt.title("Accidents by Day of Week")
plt.xlabel("Day of Week (0=Mon … 6=Sun)")
plt.ylabel("Count")
plt.show()

# 3. Severity proportions by weather
weather_sev = acc.groupby(["weather_simple", "Severity"]).size().unstack(fill_value=0)
weather_sev.div(weather_sev.sum(axis=1), axis=0).plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("Severity Proportion by Weather Condition")
plt.ylabel("Proportion")
plt.show()

# 4. Severity proportions by road surface
road_sev = acc.groupby(["road_surface", "Severity"]).size().unstack(fill_value=0)
road_sev.div(road_sev.sum(axis=1), axis=0).plot(kind="bar", stacked=True, figsize=(8, 5))
plt.title("Severity Proportion by Road Surface")
plt.ylabel("Proportion")
plt.show()

# 5. Correlation of numeric weather metrics with Severity
num_feats = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)",
    "Visibility(mi)", "Wind_Speed(mph)", "Precipitation(in)"
]
existing_feats = [f for f in num_feats if f in acc.columns]
corr = acc[existing_feats + ["Severity"]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation: Weather Metrics vs Severity")
plt.show()

# 6. Spatial heatmap of accident locations
acc2 = acc.dropna(subset=["Start_Lat", "Start_Lng"]).copy()
geometry = [Point(xy) for xy in zip(acc2["Start_Lng"], acc2["Start_Lat"])]
gdf = gpd.GeoDataFrame(acc2, geometry=geometry, crs="EPSG:4326")

m = folium.Map(
    location=[gdf["Start_Lat"].mean(), gdf["Start_Lng"].mean()],
    zoom_start=6
)
# sample up to 50k points for performance
heat_data = [
    [row["Start_Lat"], row["Start_Lng"]]
    for idx, row in gdf.sample(min(len(gdf), 50000)).iterrows()
]
HeatMap(heat_data).add_to(m)
m.save("accident_heatmap.html")
print("Saved heatmap: accident_heatmap.html")

# 7. Hourly accident proportion by weather
df = acc.groupby(["weather_simple", "hour"]).size().reset_index(name="count")
totals = acc.groupby("weather_simple").size().reset_index(name="total")
df = df.merge(totals, on="weather_simple")
df["prop"] = df["count"] / df["total"]

plt.figure(figsize=(12, 5))
for w in df["weather_simple"].unique():
    sub = df[df["weather_simple"] == w]
    plt.plot(sub["hour"], sub["prop"], label=w)
plt.title("Hourly Accident Proportion by Weather")
plt.xlabel("Hour")
plt.ylabel("Proportion")
plt.legend()
plt.show()

# 8. Chi-square test: weather vs severity
ct = pd.crosstab(acc["weather_simple"], acc["Severity"])
chi2, p, _, _ = chi2_contingency(ct)
print(f"Chi-square test: χ² = {chi2:.2f}, p-value = {p:.4f}")

# 9. Logistic regression to predict severe accidents
acc["severe_flag"] = (acc["Severity"] >= 3).astype(int)
cat_feats = ["weather_simple", "road_surface"]
num_feats2 = ["Temperature(F)", "Humidity(%)", "Visibility(mi)", "Precipitation(in)"]
existing_num_feats2 = [f for f in num_feats2 if f in acc.columns]

df_model = acc.dropna(subset=cat_feats + existing_num_feats2 + ["severe_flag"])
X = df_model[cat_feats + existing_num_feats2]
y = df_model["severe_flag"]

preproc = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_feats),
    ("num", StandardScaler(), existing_num_feats2)
])
model = make_pipeline(preproc, LogisticRegression(max_iter=200))
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)
model.fit(X_train, y_train)
print("Logistic Regression accuracy:", model.score(X_test, y_test))

# 10. DBSCAN clustering on spatial data
coords = np.vstack([gdf["Start_Lat"].values, gdf["Start_Lng"].values]).T
db = DBSCAN(eps=0.01, min_samples=20).fit(coords)
gdf["cluster"] = db.labels_

cluster_map = folium.Map(
    location=[gdf["Start_Lat"].mean(), gdf["Start_Lng"].mean()],
    zoom_start=6
)
colors = sns.color_palette("tab20", as_cmap=False).as_hex()

for idx, row in gdf.sample(min(len(gdf), 5000)).iterrows():
    c = row["cluster"]
    if c < 0:
        continue
    folium.CircleMarker(
        [row["Start_Lat"], row["Start_Lng"]],
        radius=3,
        color=colors[c % len(colors)],
        fill=True,
        fill_opacity=0.6
    ).add_to(cluster_map)

cluster_map.save("acc_clusters.html")
print("Saved clustering map: acc_clusters.html")
