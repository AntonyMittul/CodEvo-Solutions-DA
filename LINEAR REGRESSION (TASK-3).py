import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(r"C:\Users\SAM\OneDrive\Desktop\CodEvo Intersnship\cleaned_day.csv")
print(data)

#feature selection
features = data.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
target = data['cnt']
print(target)

#splitting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled)
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)

#linear regression
linear_reg = LinearRegression()
print(linear_reg.fit(X_train_scaled, y_train))

#prediction
y_pred = linear_reg.predict(X_test_scaled)
print(y_pred)

#model evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

#print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Model coefficients
coefficients = pd.DataFrame({'Feature': features.columns, 'Coefficient': linear_reg.coef_})
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print(coefficients)

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

