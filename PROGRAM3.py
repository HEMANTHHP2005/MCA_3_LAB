import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
data = pd.read_csv(r"E:\MCA_HEMANTH\AML\brain_tumor_data_101_samples.csv")
X = data[['Age', 'Headache', 'Seizure', 'VisionProblem',
          'Nausea', 'Dizziness', 'FamilyHistory']]
y = data['Tumor']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy =", round(accuracy * 100, 2), "%")
cm = confusion_matrix(y_test, y_pred)
TN = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TP = cm[1][1]
print("\nConfusion Matrix")
print(cm)
print("\nConfusion Matrix Interpretation")
print("True Negatives (No Tumor correctly predicted)      :", TN)
print("False Positives (No Tumor predicted as Tumor)      :", FP)
print("False Negatives (Tumor predicted as No Tumor)      :", FN)
print("True Positives (Tumor correctly predicted)         :", TP)
print("\nSummary")
print("Total patients in Test Set :", len(y_test))
print("Correctly identified Tumor patients     :", TP)
print("Correctly identified No Tumor patients  :", TN)
print("Incorrect Tumor predictions             :", FP + FN)
# Count predictions
tumor_count = sum(y_pred == 1)
no_tumor_count = sum(y_pred == 0)
print("\nPrediction Count")
print("Predicted Tumor Patients     :", tumor_count)
print("Predicted No Tumor Patients  :", no_tumor_count)
print("\nClassification Report")
print(classification_report(y_test, y_pred,zero_division=0))
new_patient = pd.DataFrame(
    [[35, 1, 1, 0, 1, 0, 1]],
    columns=['Age', 'Headache', 'Seizure', 'VisionProblem',
             'Nausea', 'Dizziness', 'FamilyHistory']
)
prediction = model.predict(new_patient)
probability = model.predict_proba(new_patient)
print("\nNew Patient Prediction")
print("Probability of No Tumor : {:.2f}%".format(probability[0][0] * 100))
print("Probability of Tumor    : {:.2f}%".format(probability[0][1] * 100))
if prediction[0] == 1:
    print("\nResult : Brain Tumor Detected")
else:
    print("\nResult : No Brain Tumor")
