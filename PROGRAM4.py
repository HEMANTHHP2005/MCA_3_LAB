import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Load dataset
file = r"E:\MCA_HEMANTH\AML\spam_dataset.csv"
data = pd.read_csv(file)

# Display dataset
print(data.head())

print("\nColumn Names:")
print(data.columns)

# Independent and Dependent variables
X = data["text"]
y = data["spam"]

# Convert text into numerical values
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Test with user input
email = input("\nEnter Email Message: ")

email = vectorizer.transform([email])

prediction = model.predict(email)

if prediction[0] == 1:
    print("Spam Email")
else:
    print("Not Spam Email")
