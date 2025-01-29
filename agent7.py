from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range
start_date = datetime(2018, 1, 1)
end_date = datetime(2025, 1, 1)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate synthetic credit card data with strong time series effects
n = len(date_range)

# Transaction Volume: seasonal effect + trend + noise
transaction_volume = (
    10000 + 2000 * np.sin(np.linspace(0, 50, n)) + 
    np.linspace(0, 5000, n) + np.random.normal(0, 1000, n)
).astype(int)

# Average Credit Utilization (%): smooth trend + seasonality
credit_utilization = (
    30 + 5 * np.sin(np.linspace(0, 30, n)) + 
    np.linspace(0, 10, n) + np.random.normal(0, 2, n)
)

# Delinquency Rate (%): inverse relationship with credit utilization
delinquency_rate = (
    5 + 2 * np.cos(np.linspace(0, 40, n)) - 
    0.1 * np.linspace(0, 5, n) + np.random.normal(0, 0.5, n)
)

# Macroeconomic Factor: a slow-moving index
macroeconomic_factor = (
    100 + np.cumsum(np.random.normal(0, 0.2, n))
)

# Interest Rate (%): fluctuates with macroeconomic conditions
interest_rate = (
    15 + 1.5 * np.sin(np.linspace(0, 20, n)) + 
    0.05 * macroeconomic_factor + np.random.normal(0, 0.3, n)
)

# Inflation Rate (%): steady upward trend with volatility
inflation_rate = (
    2 + 0.02 * np.linspace(0, 10, n) + 
    np.random.normal(0, 0.1, n)
)

# Create DataFrame
credit_card_data = pd.DataFrame({
    "Date": date_range,
    "Transaction Volume": transaction_volume,
    "Average Credit Utilization (%)": credit_utilization,
    "Delinquency Rate (%)": delinquency_rate,
    "Macroeconomic Factor": macroeconomic_factor,
    "Interest Rate (%)": interest_rate,
    "Inflation Rate (%)": inflation_rate
})

# Display the dataset
print(credit_card_data.head())

# Select three trending variables for prediction
features = ["Macroeconomic Factor", "Interest Rate (%)", "Inflation Rate (%)"]
target = "Transaction Volume"

# Prepare training and testing data
X = credit_card_data[features]
y = credit_card_data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train a predictive model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)

# Display model evaluation
mae
