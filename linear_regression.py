import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load the prepared CSVs (assumes you ran the previous code)
train_feat = pd.read_csv("train_features.csv", parse_dates=['date'], index_col='date')
test_feat = pd.read_csv("test_features.csv", parse_dates=['date'], index_col='date')

# Prepare X (features) and y (target) -- drop lag NaNs already handled
X_train = train_feat[['lag_1', 'lag_7']]
y_train = train_feat['sales']
X_test = test_feat[['lag_1', 'lag_7']]
y_test = test_feat['sales']

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate with metrics from earlier lessons
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot actual vs predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='Actual Sales', marker='o')
plt.plot(y_test.index, y_pred, label='Predicted Sales', marker='x')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Store 1 Grocery I Sales: Linear Regression Forecast')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
