import numpy as np, pandas as pd, yaml, os
cfg = yaml.safe_load(open("configs/default.yaml"))

# Load raw data
raw = pd.read_csv("data/raw/AT_hourly_MW_and_Price.csv", parse_dates=["Time (UTC)"])
raw = raw.set_index("Time (UTC)").sort_index()
raw = raw.loc["2017-01-01":"2022-12-31 23:00"]

# Capacity
cap = pd.read_csv("data/raw/Installed Capacity per Production Type_201701010000-202301010000.csv")
cap = cap.melt(id_vars=["Production Type"], var_name="year", value_name="MW")
cap["year"] = cap["year"].str.extract(r"(\d{4})").astype(int)
cap["ptype_norm"] = cap["Production Type"].astype(str).str.lower()
solar_cap = cap[cap["ptype_norm"].str.contains("solar")].groupby("year")["MW"].sum()
wind_cap  = cap[cap["ptype_norm"].str.contains("wind") ].groupby("year")["MW"].sum()

def annual_to_hourly(s, idx):
    y = pd.Series(index=idx, data=idx.year).map(s).astype(float)
    return y.fillna(method="ffill").fillna(method="bfill")

df = raw.copy()
df["Solar_Cap_MW"] = annual_to_hourly(solar_cap, df.index)
df["Wind_Cap_MW"]  = annual_to_hourly(wind_cap,  df.index)
df["CF_Solar"] = (df["Solar_MW"] / df["Solar_Cap_MW"]).clip(0, 1.25)
df["CF_Wind"]  = (df["Wind_MW"]  / df["Wind_Cap_MW"]).clip(0, 1.25)

# Calendar features
idx = df.index
df["hour_sin"]  = np.sin(2*np.pi*idx.hour/24); df["hour_cos"]  = np.cos(2*np.pi*idx.hour/24)
df["dow_sin"]   = np.sin(2*np.pi*idx.dayofweek/7); df["dow_cos"] = np.cos(2*np.pi*idx.dayofweek/7)
df["month_sin"] = np.sin(2*np.pi*(idx.month-1)/12); df["month_cos"] = np.cos(2*np.pi*(idx.month-1)/12)

# Derived met
T_C = df["temperature_2m (°C)"]; T_K = T_C + 273.15
p_Pa = df["surface_pressure (hPa)"]*100; RH = (df["relative_humidity_2m (%)"]/100).clip(0,1)
e_s = 610.94 * np.exp((17.625*T_C)/(T_C+243.04)); e = RH * e_s
q = 0.622 * e / (p_Pa - (1-0.622)*e).clip(lower=1.0)
rho_air = p_Pa / (287.05 * (T_K*(1+0.61*q)))
df["air_density_kgm3"] = rho_air
v100 = df["wind_speed_10m (m/s)"].clip(lower=0) * (100/10)**0.143
df["wind_speed_100m (m/s)"] = v100
df["pv_proxy"] = (df["shortwave_radiation (W/m²)"].clip(lower=0) *
                  (1.0 - 0.005*(T_C-25).clip(lower=0))).clip(lower=0)
df.rename(columns={"DA_Price_EUR_MWh":"Price_EUR_MWh"}, inplace=True)
df["wind_power_proxy"] = (df["air_density_kgm3"]*(v100**3)).clip(
    upper=np.nanpercentile((rho_air*(v100**3)).fillna(0), 99))

# Holidays & weekend flags
hol = pd.read_csv(cfg["holidays_csv"], parse_dates=["date"])
hol["date"] = hol["date"].dt.tz_convert("UTC").dt.floor("D")
df["is_public_holiday"] = df.index.floor("D").isin(hol["date"]).astype(int)
df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
df["is_special_day"] = ((df["is_public_holiday"]==1) | (df["is_weekend"]==1)).astype(int)

os.makedirs("data", exist_ok=True)
df.to_csv(cfg["engineered_csv"])
df.to_csv(cfg["engineered_full_csv"])
print(f"Saved engineered CSVs to {cfg['engineered_csv']} and {cfg['engineered_full_csv']}")
