import pandas as pd
df = pd.read_csv('train.csv')  
# --- 1️⃣ Filter and prepare the series
df1 = df[(df['store_nbr'] == 1) & (df['family'] == "GROCERY I")]
df1['date'] = pd.to_datetime(df1['date'])
df1 = df1.sort_values('date')
df1 = df1.set_index('date')

# Forward fill missing sales
sales = df1['sales'].ffill()

# --- 2️⃣ Export full series
sales.to_csv("store1_groceryI_sales.csv", header=True)
print("Exported full sales series: store1_groceryI_sales.csv")

# --- 3️⃣ Create train/test split (80/20)
train_size = int(len(sales) * 0.8)
train, test = sales[:train_size], sales[train_size:]

# Export train/test date indexes
train_dates = train.index
test_dates = test.index

train_dates.to_series().to_csv("train_dates.csv", index=False, header=False)
test_dates.to_series().to_csv("test_dates.csv", index=False, header=False)
print("Exported train/test dates: train_dates.csv, test_dates.csv")

# --- 4️⃣ Create lagged features (for ML models)
df_feat = pd.DataFrame()
df_feat['sales'] = sales
df_feat['lag_1'] = sales.shift(1)
df_feat['lag_7'] = sales.shift(7)

df_feat = df_feat.dropna()

# Split lagged features into train/test
train_dates_aligned = train_dates.intersection(df_feat.index)
df_train_feat = df_feat.loc[train_dates_aligned]
df_test_feat = df_feat.loc[test_dates]

# Export lagged features
df_train_feat.to_csv("train_features.csv", index=True)
df_test_feat.to_csv("test_features.csv", index=True)
print("Exported lagged features: train_features.csv, test_features.csv")

print("✅ All CSVs ready to provide to your teammates.")