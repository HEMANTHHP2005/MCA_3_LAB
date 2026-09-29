import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = pd.read_csv(r"E:\MCA_HEMANTH\AML\Employee.csv")
X = data[['Year_of_Experience']].values
y = data['Salary'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Intercept =", model.intercept_)
print("Slope =", model.coef_[0])
print("Accuracy (R² Score):", model.score(X_test, y_test))
salary = model.predict([[8]])
print("Predicted Salary for 8 years experience =", salary[0])
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
