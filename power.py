import pandas as pd
import numpy as np
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Step 1: Define your file paths
BAREILLY_FILES = [
    r"C:\SUJI\FinalYearProject\Power\CEEW - Smart meter data Bareilly 2020.csv",
    r"C:\SUJI\FinalYearProject\Power\CEEW - Smart meter data Bareilly 2021.csv",
    r"C:\SUJI\FinalYearProject\Power\SM Cleaned Data BR2019.csv"
]

MATHURA_FILES = [
    r"C:\SUJI\FinalYearProject\Power\CEEW - Smart meter data Mathura 2019.csv",
    r"C:\SUJI\FinalYearProject\Power\CEEW - Smart meter data Mathura 2020.csv",
    r"C:\SUJI\FinalYearProject\Power\SM Cleaned Data MH2021.csv"
]

print("🚀 Starting dataset creation...")

# Step 2: Load ALL files (handles CSV and cleaned formats automatically)
def load_smartmeter_file(file_path):
    """Load CSV or cleaned data with correct separator"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Loaded {file_path} - Shape: {df.shape}")
        return df
    except:
        # Try tab-separated if CSV fails
        df = pd.read_csv(file_path, sep='\t')
        print(f"✅ Loaded TAB {file_path} - Shape: {df.shape}")
        return df

# Load Bareilly data
bareilly_dfs = [load_smartmeter_file(f) for f in BAREILLY_FILES]
bareilly_raw = pd.concat(bareilly_dfs, ignore_index=True)

# Load Mathura data  
mathura_dfs = [load_smartmeter_file(f) for f in MATHURA_FILES]
mathura_raw = pd.concat(mathura_dfs, ignore_index=True)

# Combine both cities
raw_data = pd.concat([bareilly_raw, mathura_raw], ignore_index=True)
print(f"📊 Combined raw data: {raw_data.shape}")

# Step 3: Standardize columns (handle different naming)
print("\n🔄 Standardizing columns...")
column_mapping = {
    'x_Timestamp': ['x_Timestamp', 'Timestamp', 'timestamp'],
    'meter': ['meter', 'Meter', 'meter_id'],
    't_kWh': ['t_kWh', 'kWh', 'consumption_kwh', 'energy_kwh'],
    'z_Avg Voltage (Volt)': ['z_Avg Voltage (Volt)', 'voltage', 'Voltage'],
    'z_Avg Current (Amp)': ['z_Avg Current (Amp)', 'current', 'Current'],
    'y_Freq (Hz)': ['y_Freq (Hz)', 'frequency', 'Freq']
}

# Find actual column names
def find_column(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

data = raw_data.copy()
data['datetime'] = pd.to_datetime(data[find_column(data, column_mapping['x_Timestamp'])])
data['meter'] = data[find_column(data, column_mapping['meter'])]
data['load_kWh'] = data[find_column(data, column_mapping['t_kWh'])]

# Set datetime index and sort
data = data.set_index('datetime').sort_index()
data = data[['meter', 'load_kWh']].dropna()

print(f"✅ Standardized data: {data.shape}")

# Step 4: Aggregate to CITY-LEVEL hourly load (perfect for forecasting)
print("\n🏙️ Aggregating to city-level hourly load...")
city_hourly = data.groupby('meter')['load_kWh'].resample('H').sum().reset_index()
city_hourly = city_hourly.groupby('datetime')['load_kWh'].sum().to_frame(name='load_kWh')

# Ensure regular hourly frequency and interpolate small gaps
city_hourly = city_hourly.asfreq('H').interpolate()
print(f"✅ City hourly data: {city_hourly.shape}")

# Step 5: Add TEMPORAL FEATURES (hour, day, month, weekday, season)
print("\n⏰ Adding time features...")
df = city_hourly.copy()

df['hour'] = df.index.hour
df['day'] = df.index.day
df['month'] = df.index.month
df['weekday'] = df.index.weekday  # 0=Monday

# India-specific seasons (North India pattern)
def north_india_season(month):
    if month in [12, 1, 2]: return 0  # Winter
    if month in [3, 4, 5]:  return 1  # Summer
    if month in [6, 7, 8, 9]: return 2  # Monsoon
    return 3  # Post-monsoon

df['season'] = df['month'].apply(north_india_season)

# Step 6: Create LAG and ROLLING features (previous usage, weekly avg)
print("\n📈 Creating lag/rolling features...")
df['lag_1h'] = df['load_kWh'].shift(1)
df['lag_24h'] = df['load_kWh'].shift(24)  # Same hour yesterday
df['lag_168h'] = df['load_kWh'].shift(168)  # Same hour last week
df['rolling_24h_mean'] = df['load_kWh'].rolling(24).mean()
df['rolling_7d_mean'] = df['load_kWh'].rolling(168).mean()

# Drop rows with NaN (initial periods)
df = df.dropna()
print(f"✅ Feature-engineered data: {df.shape}")

# Step 7: Normalize (save scaler for later use)
print("\n⚖️ Normalizing data...")
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
feature_cols = ['hour', 'day', 'month', 'weekday', 'season', 'lag_1h', 'lag_24h', 
                'lag_168h', 'rolling_24h_mean', 'rolling_7d_mean']

# Fit scaler on 80% of data (train-like)
split = int(len(df) * 0.8)
scaler.fit(df[feature_cols].iloc[:split])

# Transform all data
df[feature_cols] = scaler.transform(df[feature_cols])
print("✅ Normalization complete")

# Step 8: Create CNN windows (past 24h → predict next 1h)
print("\n🧠 Creating CNN input windows...")
WINDOW_SIZE = 24  # Past 24 hours
FORECAST_HORIZON = 1  # Next 1 hour

X, y = [], []
for i in range(WINDOW_SIZE, len(df)):
    X.append(df[feature_cols].iloc[i-WINDOW_SIZE:i].values)  # (24, 10 features)
    y.append(df['load_kWh'].iloc[i])  # Target

X = np.array(X)  # Shape: (samples, 24, 10)
y = np.array(y)

print(f"✅ CNN-ready dataset:")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")

# Step 9: Train/Val/Test split (time-aware, no shuffle)
n_train = int(len(X) * 0.7)
n_val = int(len(X) * 0.15)

X_train, y_train = X[:n_train], y[:n_train]
X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Step 10: SAVE EVERYTHING
print("\n💾 Saving datasets...")

# Save raw hourly data (for dashboard)
df.reset_index().to_csv("indian_smartmeter_hourly.csv", index=False)

# Save scaler
import joblib
joblib.dump(scaler, "feature_scaler.pkl")

# Save splits as numpy arrays
np.savez_compressed("cnn_dataset.npz", 
                   X_train=X_train, y_train=y_train,
                   X_val=X_val, y_val=y_val,
                   X_test=X_test, y_test=y_test)

print("\n🎉 SUCCESS! Your CNN-ready dataset is ready:")
print("   📄 indian_smartmeter_hourly.csv  ← For dashboard/alerts")
print("   🧠 cnn_dataset.npz             ← For CNN training")
print("   ⚖️  feature_scaler.pkl         ← For inference")
print(f"   📅 Time range: {df.index.min()} to {df.index.max()}")
print(f"   ⚡ Samples: {len(X)} hourly forecasts")
