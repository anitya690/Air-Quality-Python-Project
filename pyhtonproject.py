# ============================================================
# PYTHON PROJECT: India Air Quality Analysis
# Source: Real-Time Air Quality Data — Monitoring Stations
# Dataset: project.csv (3443 records, 31 States, 268 Cities)
# Pollutants: CO, NH3, NO2, OZONE, PM10, PM2.5, SO2
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PHASE 1: LOAD & CLEAN DATA
# ============================================================

# Load CSV
df = pd.read_csv("project copy.csv")

# Strip whitespace from column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Convert last_update to datetime
df['last_update'] = pd.to_datetime(df['last_update'], dayfirst=True, errors='coerce')

# Drop rows where pollutant values are missing
df.dropna(subset=['pollutant_min', 'pollutant_max', 'pollutant_avg'], inplace=True)

# Ensure numeric types
for col in ['pollutant_min', 'pollutant_max', 'pollutant_avg', 'latitude', 'longitude']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.reset_index(drop=True, inplace=True)

print("\n✅ Data Loaded & Cleaned!")
print(f"Shape after cleaning : {df.shape}")
print(f"States               : {df['state'].nunique()}")
print(f"Cities               : {df['city'].nunique()}")
print(f"Pollutants           : {sorted(df['pollutant_id'].unique())}")
print("\nFirst 5 rows:")
print(df[['state', 'city', 'pollutant_id', 'pollutant_avg']].head())
print("\nBasic Statistics:")
print(df[['pollutant_min', 'pollutant_max', 'pollutant_avg']].describe())


# ============================================================
# PHASE 2: EDA — 5 OBJECTIVES
# ============================================================

# ============================================================
# OBJECTIVE 1: Average Pollution Level per Pollutant (National)
# ============================================================
pollutant_avg = df.groupby('pollutant_id')['pollutant_avg'].mean().reset_index()
pollutant_avg.columns = ['Pollutant', 'Avg_Level']
pollutant_avg = pollutant_avg.sort_values('Avg_Level', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 1: National Average Pollution Level per Pollutant")
print("=" * 55)
print(pollutant_avg.to_string(index=False))

# ============================================================
# OBJECTIVE 2: Top 10 & Bottom 5 Most Polluted States
#              (by average pollutant_avg across all pollutants)
# ============================================================
state_pollution = df.groupby('state')['pollutant_avg'].mean().reset_index()
state_pollution.columns = ['State', 'Avg_Pollution']
state_pollution = state_pollution.sort_values('Avg_Pollution', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 2: Top 10 Most Polluted States")
print("=" * 55)
print(state_pollution.head(10).to_string(index=False))
print("\nBottom 5 Least Polluted States:")
print(state_pollution.tail(5).to_string(index=False))

# ============================================================
# OBJECTIVE 3: Pollutant Spread — Min vs Max vs Avg
# ============================================================
spread = df.groupby('pollutant_id').agg(
    Min=('pollutant_min', 'mean'),
    Avg=('pollutant_avg', 'mean'),
    Max=('pollutant_max', 'mean')
).reset_index()
spread.columns = ['Pollutant', 'Avg_Min', 'Avg_Mean', 'Avg_Max']
spread['Range'] = (spread['Avg_Max'] - spread['Avg_Min']).round(2)
spread = spread.sort_values('Range', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 3: Pollutant Spread (Min / Avg / Max)")
print("=" * 55)
print(spread.to_string(index=False))

# ============================================================
# OBJECTIVE 4: City-level Pollution Hotspots
#              Top 10 most polluted cities (avg across all pollutants)
# ============================================================
city_pollution = df.groupby(['state', 'city'])['pollutant_avg'].mean().reset_index()
city_pollution.columns = ['State', 'City', 'Avg_Pollution']
city_pollution = city_pollution.sort_values('Avg_Pollution', ascending=False)

print("\n" + "=" * 55)
print("OBJECTIVE 4: Top 10 Most Polluted Cities (Hotspots)")
print("=" * 55)
print(city_pollution.head(10).to_string(index=False))

# ============================================================
# OBJECTIVE 5: Pollutant Correlation — PM2.5 vs PM10
# ============================================================
pm25 = df[df['pollutant_id'] == 'PM2.5'][['city', 'pollutant_avg']].rename(
    columns={'pollutant_avg': 'PM2.5'})
pm10 = df[df['pollutant_id'] == 'PM10'][['city', 'pollutant_avg']].rename(
    columns={'pollutant_avg': 'PM10'})
corr_df = pd.merge(pm25, pm10, on='city')
correlation = corr_df[['PM2.5', 'PM10']].corr().iloc[0, 1]

print("\n" + "=" * 55)
print("OBJECTIVE 5: PM2.5 vs PM10 Correlation")
print("=" * 55)
print(f"  Cities with both readings : {len(corr_df)}")
print(f"  Pearson Correlation       : {correlation:.4f}")
print(corr_df.head(10).to_string(index=False))


# ============================================================
# PHASE 3: VISUALISATIONS
# ============================================================

sns.set_style("whitegrid")

# ----------------------------------------------------------
# PLOT 1 (Obj 1): Average Pollution per Pollutant — Bar Chart
# ----------------------------------------------------------
plt.figure(figsize=(10, 5))
bars = plt.bar(pollutant_avg['Pollutant'], pollutant_avg['Avg_Level'],
               color='steelblue', edgecolor='black', linewidth=0.6)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 1,
             f"{bar.get_height():.1f}",
             ha='center', fontsize=9, fontweight='bold')
plt.title("Objective 1: National Average Pollution Level per Pollutant",
          fontsize=13, fontweight='bold')
plt.xlabel("Pollutant")
plt.ylabel("Average Level (µg/m³ or ppb)")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 2 (Obj 2): Top 10 Polluted States — Horizontal Bar
# ----------------------------------------------------------
top10 = state_pollution.head(10)
plt.figure(figsize=(10, 6))
bars = plt.barh(top10['State'][::-1], top10['Avg_Pollution'][::-1], color='coral')
plt.title("Objective 2: Top 10 Most Polluted States (Avg Across All Pollutants)",
          fontsize=12, fontweight='bold')
plt.xlabel("Average Pollutant Level")
for bar in bars:
    plt.text(bar.get_width() + 0.5,
             bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.1f}",
             va='center', fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 3 (Obj 3): Pollutant Min/Avg/Max — Grouped Bar Chart
# ----------------------------------------------------------
x = np.arange(len(spread))
width = 0.25
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, spread['Avg_Min'], width, label='Avg Min', color='skyblue', edgecolor='black')
ax.bar(x,         spread['Avg_Mean'], width, label='Avg Mean', color='steelblue', edgecolor='black')
ax.bar(x + width, spread['Avg_Max'], width, label='Avg Max', color='navy', edgecolor='black')
ax.set_xticks(x)
ax.set_xticklabels(spread['Pollutant'])
ax.set_title("Objective 3: Pollutant Spread — Min / Avg / Max per Pollutant",
             fontsize=13, fontweight='bold')
ax.set_ylabel("Concentration Level")
ax.legend()
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 4 (Obj 4): Top 10 Polluted Cities — Horizontal Bar
# ----------------------------------------------------------
top10_cities = city_pollution.head(10).copy()
top10_cities['Label'] = top10_cities['City'] + " (" + top10_cities['State'] + ")"
plt.figure(figsize=(12, 6))
bars = plt.barh(top10_cities['Label'][::-1],
                top10_cities['Avg_Pollution'][::-1], color='tomato')
plt.title("Objective 4: Top 10 Most Polluted Cities (Hotspots)",
          fontsize=13, fontweight='bold')
plt.xlabel("Average Pollutant Level")
for bar in bars:
    plt.text(bar.get_width() + 0.5,
             bar.get_y() + bar.get_height() / 2,
             f"{bar.get_width():.1f}",
             va='center', fontsize=8)
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# PLOT 5 (Obj 5): PM2.5 vs PM10 Scatter + Heatmap
# ----------------------------------------------------------

# Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(corr_df['PM2.5'], corr_df['PM10'],
            color='purple', alpha=0.6, edgecolor='black', s=60)
m, b = np.polyfit(corr_df['PM2.5'], corr_df['PM10'], 1)
x_line = np.linspace(corr_df['PM2.5'].min(), corr_df['PM2.5'].max(), 100)
plt.plot(x_line, m * x_line + b, color='red', linestyle='--', linewidth=2,
         label=f"Trend line (r={correlation:.2f})")
plt.title("Objective 5: PM2.5 vs PM10 Correlation Across Cities",
          fontsize=13, fontweight='bold')
plt.xlabel("PM2.5 Average Level")
plt.ylabel("PM10 Average Level")
plt.legend()
plt.tight_layout()
plt.show()

# State × Pollutant Heatmap
top15_states = state_pollution.head(15)['State'].tolist()
heatmap_data = df[df['state'].isin(top15_states)].groupby(
    ['state', 'pollutant_id'])['pollutant_avg'].mean().reset_index()
heatmap_pivot = heatmap_data.pivot(index='state', columns='pollutant_id', values='pollutant_avg')

plt.figure(figsize=(13, 8))
sns.heatmap(heatmap_pivot, cmap='YlOrRd', linewidths=0.4,
            annot=True, fmt='.0f', annot_kws={'size': 8})
plt.title("Pollution Heatmap — Top 15 States × Pollutant",
          fontsize=13, fontweight='bold')
plt.xlabel("Pollutant")
plt.ylabel("State")
plt.tight_layout()
plt.show()


# ============================================================
# PHASE 4: LINEAR REGRESSION MODEL
# — Predict PM2.5 average from pollutant_min
# ============================================================
print("\n" + "=" * 55)
print("PHASE 4: LINEAR REGRESSION MODEL")
print("— Feature: PM2.5 Min  |  Target: PM2.5 Avg")
print("=" * 55)

pm25_df = df[df['pollutant_id'] == 'PM2.5'][['pollutant_min', 'pollutant_avg']].dropna()

X = pm25_df[['pollutant_min']]
y = pm25_df['pollutant_avg']

# Train / Test split (80/20)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Model Equation  : PM2.5_avg = {model.coef_[0]:.4f} × PM2.5_min + {model.intercept_:.4f}")
print(f"Interpretation  : For every 1 unit rise in PM2.5 min, avg rises by ~{model.coef_[0]:.2f}")

# Cross-Validation
cv_scores = cross_val_score(LinearRegression(), X, y, cv=5, scoring='r2')
print(f"\nCross-Validated R² scores : {[round(s, 4) for s in cv_scores]}")
print(f"Mean CV R²                : {cv_scores.mean():.4f}")

# Test set evaluation
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n📐 Model Accuracy Metrics (Test Set):")
print(f"  MAE  : {mae:.2f}  ← avg prediction error in µg/m³")
print(f"  MSE  : {mse:.2f}")
print(f"  RMSE : {rmse:.2f}")
print(f"  R²   : {r2:.4f}  ← model explains {r2:.1%} of variance")

# Actual vs Predicted Table (first 15 test samples)
print("\nActual vs Predicted (Sample — first 15 from test set):")
print(f"{'#':<5} {'Actual PM2.5 Avg':>18} {'Predicted':>12} {'Error':>10}")
print("-" * 47)
for i, (act, pred) in enumerate(zip(list(y_test)[:15], list(y_pred)[:15]), 1):
    print(f"{i:<5} {act:>18.1f} {pred:>12.1f} {act - pred:>+10.1f}")

# Future Projections — predict avg PM2.5 for given min levels
future_mins = pd.DataFrame({'pollutant_min': [10, 20, 30, 40, 50, 60, 80, 100, 120, 150]})
future_pred = model.predict(future_mins)

print("\n🔮 PM2.5 Average Projections for Given Min Levels:")
print(f"{'PM2.5_Min':>12} {'Projected_Avg':>15}")
print("-" * 28)
for mn, pr in zip(future_mins['pollutant_min'], future_pred):
    print(f"{mn:>12} {pr:>15.1f}")

# Regression Plot
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test,
            color='steelblue', s=60, alpha=0.6, zorder=5, label='Actual (Test Set)')
x_range = pd.DataFrame({'pollutant_min': np.linspace(X['pollutant_min'].min(), X['pollutant_min'].max(), 200)})
plt.plot(x_range, model.predict(x_range),
         color='tomato', linewidth=2.5, linestyle='--', label='Regression Line')
plt.scatter(future_mins['pollutant_min'], future_pred,
            color='orange', s=90, zorder=6, marker='D', label='Projected Points')
plt.title("Linear Regression: PM2.5 Min → PM2.5 Avg Prediction",
          fontsize=14, fontweight='bold')
plt.xlabel("PM2.5 Min (µg/m³)")
plt.ylabel("PM2.5 Avg (µg/m³)")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

print("\n✅ PROJECT COMPLETE!")


# ============================================================
# PHASE 5: MODEL VALIDATION — Pollutant-wise R² Comparison
# How well does a Min→Avg linear model work across pollutants?
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: MODEL VALIDATION — Per-Pollutant Regression Check")
print("=" * 60)

validation_results = []

for pollutant in sorted(df['pollutant_id'].unique()):
    sub = df[df['pollutant_id'] == pollutant][['pollutant_min', 'pollutant_avg']].dropna()
    if len(sub) < 20:
        continue

    X_p = sub[['pollutant_min']]
    y_p = sub['pollutant_avg']

    X_tr, X_te, y_tr, y_te = train_test_split(X_p, y_p, test_size=0.2, random_state=42)
    m = LinearRegression().fit(X_tr, y_tr)
    y_pr = m.predict(X_te)

    r2_val  = r2_score(y_te, y_pr)
    mae_val = mean_absolute_error(y_te, y_pr)
    rmse_val = np.sqrt(mean_squared_error(y_te, y_pr))

    validation_results.append({
        'Pollutant': pollutant,
        'Samples': len(sub),
        'Coeff': round(m.coef_[0], 4),
        'Intercept': round(m.intercept_, 4),
        'MAE': round(mae_val, 2),
        'RMSE': round(rmse_val, 2),
        'R²': round(r2_val, 4)
    })
    print(f"  ✅ {pollutant:<8} — R²: {r2_val:.4f}  |  MAE: {mae_val:.2f}  |  RMSE: {rmse_val:.2f}")

val_df = pd.DataFrame(validation_results)

print(f"\n{'Pollutant':<12} {'Samples':>8} {'Coeff':>10} {'Intercept':>12} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-" * 72)
for _, row in val_df.iterrows():
    print(f"{row['Pollutant']:<12} {row['Samples']:>8} {row['Coeff']:>10} "
          f"{row['Intercept']:>12} {row['MAE']:>8} {row['RMSE']:>8} {row['R²']:>8}")

# ── Validation Bar Chart — R² per Pollutant ─────────────────
colors_r2 = ['green' if r >= 0.7 else 'orange' if r >= 0.5 else 'red'
              for r in val_df['R²']]

plt.figure(figsize=(10, 5))
plt.bar(val_df['Pollutant'], val_df['R²'], color=colors_r2, edgecolor='black', linewidth=0.6)
plt.axhline(0.7, color='green', linestyle='--', linewidth=1.5, label='Good fit threshold (R²=0.7)')
plt.axhline(0.5, color='orange', linestyle='--', linewidth=1.5, label='Acceptable threshold (R²=0.5)')
for i, (pol, r2v) in enumerate(zip(val_df['Pollutant'], val_df['R²'])):
    plt.text(i, r2v + 0.01, f"{r2v:.2f}", ha='center', fontsize=9, fontweight='bold')
plt.title("Phase 5 Validation: Min→Avg Linear Regression R² per Pollutant",
          fontsize=13, fontweight='bold')
plt.xlabel("Pollutant")
plt.ylabel("R² Score")
plt.ylim(0, 1.1)
plt.legend()
plt.tight_layout()
plt.show()

print("\n✅ VALIDATION COMPLETE!")
