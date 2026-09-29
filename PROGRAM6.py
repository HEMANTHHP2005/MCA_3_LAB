# VEHICLE CLASSIFICATION USING DECISION TREE

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------
# 1. ASK CSV FILE PATH
# ------------------------------------------------------------

file_path = input("Enter the CSV file path: ")

data = pd.read_csv(file_path)

print("\nCSV file loaded successfully!")
print("Number of records:", len(data))

# ------------------------------------------------------------
# 2. REMOVE EMPTY VALUES
# ------------------------------------------------------------

data = data.dropna()

# Remove unwanted spaces from column names
data.columns = data.columns.str.strip()

print("\nColumn names:")
print(data.columns.tolist())

# ------------------------------------------------------------
# 3. SELECT INPUT AND OUTPUT
# ------------------------------------------------------------

X = data[
    [
        "Year",
        "Month_Name",
        "State",
        "Vehicle_Category",
        "Vehicle_Type",
        "EV_Sales_Quantity"
    ]
].copy()

y = data["Vehicle_Class"].copy()

# ------------------------------------------------------------
# 4. CLEAN TEXT DATA
# ------------------------------------------------------------

text_columns = [
    "Month_Name",
    "State",
    "Vehicle_Category",
    "Vehicle_Type"
]

for column in text_columns:
    X[column] = (
        X[column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

y = y.astype(str).str.strip()

# ------------------------------------------------------------
# 5. CONVERT NUMERIC COLUMNS
# ------------------------------------------------------------

X["Year"] = pd.to_numeric(X["Year"], errors="coerce")
X["EV_Sales_Quantity"] = pd.to_numeric(
    X["EV_Sales_Quantity"],
    errors="coerce"
)

# Remove rows that became empty
valid_rows = X.notna().all(axis=1) & y.notna()

X = X[valid_rows]
y = y[valid_rows]

# ------------------------------------------------------------
# 6. ENCODE TEXT DATA
# ------------------------------------------------------------

encoders = {}

for column in text_columns:

    encoder = LabelEncoder()

    X[column] = encoder.fit_transform(X[column])

    encoders[column] = encoder

# ------------------------------------------------------------
# 7. ENCODE VEHICLE CLASS
# ------------------------------------------------------------

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

# ------------------------------------------------------------
# 8. CHECK DATA TYPES
# ------------------------------------------------------------

print("\nData types after encoding:")
print(X.dtypes)

print("\nFirst 5 rows after encoding:")
print(X.head())

# ------------------------------------------------------------
# 9. CREATE DECISION TREE
# ------------------------------------------------------------

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5,
    random_state=42
)

model.fit(X, y)

print("\nDecision Tree model trained successfully!")

# ------------------------------------------------------------
# 10. USER INPUT
# ------------------------------------------------------------

print("\n==============================================")
print("        ENTER VEHICLE DETAILS")
print("==============================================")

year = int(input("Enter Year: "))

month = input("Enter Month Name: ").strip().lower()

state = input("Enter State: ").strip().lower()

category = input("Enter Vehicle Category: ").strip().lower()

vehicle_type = input("Enter Vehicle Type: ").strip().lower()

ev_sales = int(input("Enter EV Sales Quantity: "))

# ------------------------------------------------------------
# 11. CHECK USER INPUT
# ------------------------------------------------------------

if month not in encoders["Month_Name"].classes_:
    print("\nInvalid Month Name!")
    print("Available months:")
    print(list(encoders["Month_Name"].classes_))
    exit()

if state not in encoders["State"].classes_:
    print("\nInvalid State!")
    print("Available states:")
    print(list(encoders["State"].classes_))
    exit()

if category not in encoders["Vehicle_Category"].classes_:
    print("\nInvalid Vehicle Category!")
    print("Available categories:")
    print(list(encoders["Vehicle_Category"].classes_))
    exit()

if vehicle_type not in encoders["Vehicle_Type"].classes_:
    print("\nInvalid Vehicle Type!")
    print("Available vehicle types:")
    print(list(encoders["Vehicle_Type"].classes_))
    exit()

# ------------------------------------------------------------
# 12. ENCODE USER INPUT
# ------------------------------------------------------------

month_encoded = encoders["Month_Name"].transform([month])[0]

state_encoded = encoders["State"].transform([state])[0]

category_encoded = encoders["Vehicle_Category"].transform([category])[0]

vehicle_type_encoded = encoders["Vehicle_Type"].transform([vehicle_type])[0]

# ------------------------------------------------------------
# 13. CREATE INPUT DATA
# ------------------------------------------------------------

input_data = pd.DataFrame(
    [
        [
            year,
            month_encoded,
            state_encoded,
            category_encoded,
            vehicle_type_encoded,
            ev_sales
        ]
    ],
    columns=[
        "Year",
        "Month_Name",
        "State",
        "Vehicle_Category",
        "Vehicle_Type",
        "EV_Sales_Quantity"
    ]
)

# ------------------------------------------------------------
# 14. PREDICT VEHICLE CLASS
# ------------------------------------------------------------

prediction = model.predict(input_data)

predicted_vehicle = target_encoder.inverse_transform(prediction)

# ------------------------------------------------------------
# 15. DISPLAY RESULT
# ------------------------------------------------------------

print("\n==============================================")
print("       VEHICLE CLASSIFICATION RESULT")
print("==============================================")

print("Predicted Vehicle Class:", predicted_vehicle[0])

print("==============================================")
