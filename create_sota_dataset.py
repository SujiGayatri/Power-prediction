import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("🔗 Creating SOTA dataset with weather...")

# Load base + weather
df = pd.read_csv("indian_smartmeter_hourly.csv", parse_dates=['datetime']).set_index('datetime')
weather = pd.read_csv("up_weather_hourly_2019_2021.csv", parse_dates=['datetime']).set_index('datetime')

# Merge perfectly aligned data
df_sota = df.join(weather[['temperature', 'humidity', 'rain']], how='inner')
print(f"✅ Merged: {len(df)} → {len(df_sota)} rows (+ weather!)")

# SOTA Feature Engineering (13 features total)
df_sota['hour'] = df_sota.index.hour
df_sota['day'] = df_sota.index.day
df_sota['month'] = df_sota.index.month
df_sota['weekday'] = df_sota.index.weekday

def season_map(m):
    if m in [12,1,2]: return 0  # Winter
    if m in [3,4,5]: return 1   # Summer
    if m in [6,7,8,9]: return 2 # Monsoon
    return 3                    # Post-monsoon

df_sota['season'] = df_sota['month'].apply(season_map)

# Lag + Rolling features
df_sota['lag_1h'] = df_sota['load_kWh'].shift(1)
df_sota['lag_24h'] = df_sota['load_kWh'].shift(24)
df_sota['lag_168h'] = df_sota['load_kWh'].shift(168)
df_sota['rolling_24h'] = df_sota['load_kWh'].rolling(24).mean()
df_sota['rolling_7d'] = df_sota['load_kWh'].rolling(168).mean()

df_sota = df_sota.dropna()
print(f"✅ Feature engineered: {len(df_sota)} rows")

# **SOTA 13-Feature Set**
feature_cols = ['hour', 'day', 'month', 'weekday', 'season', 'lag_1h', 'lag_24h', 
                'lag_168h', 'rolling_24h', 'rolling_7d', 'temperature', 'humidity', 'rain']

# Normalize (train stats only)
scaler = StandardScaler()
split = int(0.8 * len(df_sota))
scaler.fit(df_sota[feature_cols].iloc[:split])
df_sota[feature_cols] = scaler.transform(df_sota[feature_cols])
joblib.dump(scaler, "sota_scaler.pkl")

# CNN Windows: Past 24h → Next 1h (24 timesteps × 13 features)
WINDOW = 24
X, y = [], []
for i in range(WINDOW, len(df_sota)):
    X.append(df_sota[feature_cols].iloc[i-WINDOW:i].values)  # (24, 13)
    y.append(df_sota['load_kWh'].iloc[i])

X, y = np.array(X), np.array(y)
print(f"✅ SOTA CNN shape: {X.shape}")

# Time-aware splits
n_train = int(0.7 * len(X))
n_val = int(0.85 * len(X))
np.savez_compressed("sota_dataset.npz",
                   X_train=X[:n_train], y_train=y[:n_train],
                   X_val=X[n_train:n_val], y_val=y[n_train:n_val],
                   X_test=X[n_val:], y_test=y[n_val:])

df_sota.reset_index().to_csv("sota_smartmeter_hourly.csv", index=False)

print("🏆 SOTA DATASET READY!")
print(f"   📊 Shape: {X.shape} (24h × 13 features)")
print("   💾 Files: sota_dataset.npz + sota_scaler.pkl")
