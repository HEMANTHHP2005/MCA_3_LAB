import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# Get dataset path
dataset_path = input("Enter weather dataset path: ")
# Read dataset
data = pd.read_csv(dataset_path)
# Convert original Summary into 3 classes
def weather_class(summary):
    summary = str(summary).lower()
    if "rain" in summary or "drizzle" in summary:
        return "Rainy"
    elif "clear" in summary or "dry" in summary:
        return "Clear"
    else:
        return "Cloudy"
data["Weather_Class"] = data["Summary"].apply(weather_class)
# Select input features
features = [
    "Temperature (C)",
    "Apparent Temperature (C)",
    "Humidity",
    "Wind Speed (km/h)",
    "Wind Bearing (degrees)",
    "Visibility (km)",
    "Pressure (millibars)"
]
# Remove missing values
data = data.dropna(subset=features + ["Weather_Class"])
X = data[features]
y = data["Weather_Class"]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
# Scaling for KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
# Naive Bayes model
nb = GaussianNB()
nb.fit(X_train, y_train)
# Get new weather input
temperature = float(input("Enter Temperature (C): "))
apparent_temperature = float(input("Enter Apparent Temperature (C): "))
humidity = float(input("Enter Humidity: "))
wind_speed = float(input("Enter Wind Speed (km/h): "))
wind_bearing = float(input("Enter Wind Bearing (degrees): "))
visibility = float(input("Enter Visibility (km): "))
pressure = float(input("Enter Pressure (millibars): "))
# Create input data
new_weather = pd.DataFrame([[
    temperature,
    apparent_temperature,
    humidity,
    wind_speed,
    wind_bearing,
    visibility,
    pressure
]], columns=features)
# KNN prediction
new_weather_scaled = scaler.transform(new_weather)
knn_result = knn.predict(new_weather_scaled)[0]
# Naive Bayes prediction
nb_result = nb.predict(new_weather)[0]
# Display only results
print("\nKNN Result       :", knn_result)
print("Naive Bayes Result:", nb_result)
