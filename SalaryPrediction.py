# 📘 Assignment 6: Salary Prediction using Linear Regression

# ✅ 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ✅ 2. Load and Explore the Dataset
df = pd.read_csv('Salary_Data.csv')
print("📊 First 5 Rows of the Dataset:")
print(df.head())

print("\n📈 Dataset Description:")
print(df.describe())

print("\nℹ️ Dataset Info:")
print(df.info())

# ✅ 3. Visualize the Data
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='YearsExperience', y='Salary', color='blue')
plt.title("Salary vs Years of Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.grid(True)
plt.show()

# ✅ 4. Split the Dataset into Training and Testing Sets
X = df[['YearsExperience']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ 5. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# ✅ 6. Predict Salaries for the Test Set
y_pred = model.predict(X_test)

# ✅ 7. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n📉 Mean Squared Error (MSE): {mse:.2f}")
print(f"📊 R² Score: {r2:.2f}")

# ✅ 8. Plot the Regression Line
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X['YearsExperience'], y=y, label="Actual Data")
plt.plot(X, model.predict(X), color='red', label="Regression Line")
plt.title("Linear Regression Fit")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()

# ✅ BONUS A: Error Bars to Show Prediction Confidence
errors = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.errorbar(X_test.values.flatten(), y_test, yerr=np.abs(errors), fmt='o', label="Prediction Errors", color='purple')
plt.plot(X, model.predict(X), color='green', label="Regression Line")
plt.title("Prediction Errors with Error Bars")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.show()

# ✅ BONUS B: Bar Chart Comparing Actual vs Predicted
comparison_df = pd.DataFrame({'Actual Salary': y_test.values, 'Predicted Salary': y_pred})
comparison_df.reset_index(drop=True, inplace=True)
comparison_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
plt.title("Actual vs Predicted Salaries")
plt.xlabel("Test Sample Index")
plt.ylabel("Salary")
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ BONUS C: Predict Salary from Custom User Input
try:
    custom_exp = float(input("🧮 Enter years of experience to predict salary: "))
    custom_pred = model.predict([[custom_exp]])
    print(f"💰 Predicted Salary for {custom_exp} years of experience is ₹{custom_pred[0]:,.2f}")
except ValueError:
    print("❌ Invalid input. Please enter a numeric value.")
