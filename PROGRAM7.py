import csv

# ============================================================
# K-MEANS CREDIT CARD FRAUD DETECTION
# Simple Python Program
# ============================================================

print("==============================================")
print("   CREDIT CARD FRAUD DETECTION USING K-MEANS")
print("==============================================")

# ------------------------------------------------------------
# 1. GET CSV FILE PATH
# ------------------------------------------------------------

file_path = input("\nEnter the CSV file path: ")

# ------------------------------------------------------------
# 2. READ DATASET
# ------------------------------------------------------------

amounts = []
actual_class = []
rows = []

with open(file_path, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        amount = float(row["Amount"])

        amounts.append(amount)
        rows.append(row)

        if "Class" in row:
            actual_class.append(int(row["Class"]))

print("\nDataset loaded successfully.")

print("Total transactions:", len(amounts))

# ------------------------------------------------------------
# 3. ASK INITIAL CENTROID VALUES
# ------------------------------------------------------------

print("\nEnter the initial centroid values.")

centroid1 = float(
    input("Enter Centroid 1 value: ")
)

centroid2 = float(
    input("Enter Centroid 2 value: ")
)

print("\nInitial Centroids:")
print("Centroid 1 =", centroid1)
print("Centroid 2 =", centroid2)

# ------------------------------------------------------------
# 4. K-MEANS ITERATION
# ------------------------------------------------------------

for iteration in range(10):

    cluster1 = []
    cluster2 = []

    # --------------------------------------------------------
    # ASSIGN EACH TRANSACTION TO NEAREST CENTROID
    # --------------------------------------------------------

    for amount in amounts:

        distance1 = abs(amount - centroid1)
        distance2 = abs(amount - centroid2)

        if distance1 <= distance2:
            cluster1.append(amount)

        else:
            cluster2.append(amount)

    # --------------------------------------------------------
    # CALCULATE NEW CENTROIDS
    # --------------------------------------------------------

    if len(cluster1) > 0:

        new_centroid1 = sum(cluster1) / len(cluster1)

    else:

        new_centroid1 = centroid1

    if len(cluster2) > 0:

        new_centroid2 = sum(cluster2) / len(cluster2)

    else:

        new_centroid2 = centroid2

    print("\nIteration", iteration + 1)

    print("Centroid 1:", round(new_centroid1, 2))
    print("Centroid 2:", round(new_centroid2, 2))

    # --------------------------------------------------------
    # CHECK WHETHER CENTROIDS HAVE STOPPED CHANGING
    # --------------------------------------------------------

    if (
        abs(new_centroid1 - centroid1) < 0.01
        and
        abs(new_centroid2 - centroid2) < 0.01
    ):

        centroid1 = new_centroid1
        centroid2 = new_centroid2

        break

    centroid1 = new_centroid1
    centroid2 = new_centroid2

# ------------------------------------------------------------
# 5. DISPLAY FINAL CENTROIDS
# ------------------------------------------------------------

print("\n==============================================")
print("             FINAL CENTROIDS")
print("==============================================")

print("Final Centroid 1:", round(centroid1, 2))
print("Final Centroid 2:", round(centroid2, 2))

# ------------------------------------------------------------
# 6. IDENTIFY HIGH-VALUE CLUSTER
# ------------------------------------------------------------

if centroid1 > centroid2:

    fraud_centroid = centroid1
    normal_centroid = centroid2

else:

    fraud_centroid = centroid2
    normal_centroid = centroid1

print("\nNormal Cluster Centroid:",
      round(normal_centroid, 2))

print("Suspicious Cluster Centroid:",
      round(fraud_centroid, 2))

# ------------------------------------------------------------
# 7. FIND POSSIBLE FRAUD TRANSACTIONS
# ------------------------------------------------------------

fraud_transactions = []

normal_transactions = []

for i in range(len(amounts)):

    amount = amounts[i]

    distance1 = abs(amount - centroid1)
    distance2 = abs(amount - centroid2)

    if distance1 <= distance2:

        assigned_centroid = centroid1

    else:

        assigned_centroid = centroid2

    # Higher centroid is considered suspicious
    if assigned_centroid == fraud_centroid:

        fraud_transactions.append(i)

    else:

        normal_transactions.append(i)

# ------------------------------------------------------------
# 8. DISPLAY FRAUD TRANSACTIONS
# ------------------------------------------------------------

print("\n==============================================")
print("       POSSIBLE FRAUD TRANSACTIONS")
print("==============================================")

print("Number of possible fraud transactions:",
      len(fraud_transactions))

print()

for i in fraud_transactions[:50]:

    print(
        "Transaction", i + 1,
        "| Amount =", amounts[i]
    )

# ------------------------------------------------------------
# 9. DISPLAY ACTUAL FRAUD INFORMATION
# ------------------------------------------------------------

if len(actual_class) > 0:

    actual_fraud = 0
    detected_actual_fraud = 0

    for i in range(len(actual_class)):

        if actual_class[i] == 1:

            actual_fraud = actual_fraud + 1

            if i in fraud_transactions:

                detected_actual_fraud = (
                    detected_actual_fraud + 1
                )

    print("\n==============================================")
    print("           ACTUAL FRAUD ANALYSIS")
    print("==============================================")

    print("Actual fraud transactions:",
          actual_fraud)

    print("Fraud transactions identified by K-means:",
          detected_actual_fraud)

    if actual_fraud > 0:

        detection_rate = (
            detected_actual_fraud / actual_fraud
        ) * 100

        print(
            "Fraud detection rate:",
            round(detection_rate, 2),
            "%"
        )

print("\n==============================================")
print("             ANALYSIS COMPLETED")
print("==============================================")
