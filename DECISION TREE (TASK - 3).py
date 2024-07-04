import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# Load the dataset
data = pd.read_csv(r"C:\Users\SAM\OneDrive\Desktop\CodEvo Intersnship\cleaned_day.csv")
print(data)

# Define features and target variable
features = data.drop(columns=['instant', 'dteday', 'casual', 'registered', 'cnt'])
target = data['cnt']
print(target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
print(X_train_scaled)
X_test_scaled = scaler.transform(X_test)
print(X_test_scaled)

# Train the Decision Tree Regressor model
decision_tree = DecisionTreeRegressor(random_state=42)
print(decision_tree.fit(X_train_scaled, y_train))

# Make predictions on the test set
y_pred = decision_tree.predict(X_test_scaled)
print(y_pred)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(decision_tree, feature_names=features.columns, filled=True, rounded=True)
plt.show()

# Perform cross-validation
cv_scores = cross_val_score(decision_tree, X_train_scaled, y_train, cv=5, scoring='r2')
print("Cross-Validation R2 Scores:", cv_scores)
print("Mean Cross-Validation R2 Score:", cv_scores.mean())

