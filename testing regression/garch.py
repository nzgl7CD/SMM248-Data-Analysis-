import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# Simulate some return data (for example, from a normal distribution)
np.random.seed(42)
returns = np.random.normal(0, 1, 1000)  # 1000 observations of return
returns = pd.Series(returns)

# Create a GARCH model with 1 lag for both the squared error term and volatility
model = arch_model(returns, vol='Garch', p=1, q=1)

# Fit the model
model_fitted = model.fit()

print(model_fitted.summary())

# Forecast the next 5 steps of volatility
forecast = model_fitted.forecast(horizon=5)

print(forecast.variance[-1:])

# Plot the conditional volatility (standard deviation)
plt.figure(figsize=(10,6))
plt.plot(model_fitted.conditional_volatility, label='Conditional Volatility')
plt.title('Conditional Volatility from GARCH(1, 1) Model')
plt.legend()
plt.show()
