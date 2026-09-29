import tensorflow as tf
import numpy as np
import os

# ============================================================
# DODDAPATHRE DISEASE CLASSIFICATION USING CNN
# Dynamic folder path + Classification Report
# ============================================================

# ------------------------------------------------------------
# 1. GET DATASET PATHS DYNAMICALLY
# ------------------------------------------------------------

train_path = input("Enter the path of TRAIN folder: ")
test_path = input("Enter the path of TEST folder: ")

# Remove quotation marks if the user copies the path from
# Windows File Explorer
train_path = train_path.strip('"')
test_path = test_path.strip('"')

# ------------------------------------------------------------
# 2. CHECK PATHS
# ------------------------------------------------------------

if not os.path.exists(train_path):
    print("\nTRAIN folder does not exist.")
    print("Please check the path.")
    exit()

if not os.path.exists(test_path):
    print("\nTEST folder does not exist.")
    print("Please check the path.")
    exit()

print("\nTrain folder found successfully.")
print("Test folder found successfully.")

# ------------------------------------------------------------
# 3. SETTINGS
# ------------------------------------------------------------

IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 1

# ------------------------------------------------------------
# 4. LOAD TRAINING DATA
# ------------------------------------------------------------

print("\nLoading training images...")

train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ------------------------------------------------------------
# 5. LOAD TEST DATA
# ------------------------------------------------------------

print("\nLoading testing images...")

test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ------------------------------------------------------------
# 6. DISPLAY CLASSES
# ------------------------------------------------------------

class_names = train_data.class_names

print("\nClasses found:")
print("--------------------------------")

for i in range(len(class_names)):
    print(i, "=", class_names[i])

# ------------------------------------------------------------
# 7. CNN MODEL
# ------------------------------------------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Rescaling(
        1.0 / 255,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(
        len(class_names),
        activation="softmax"
    )
])

# ------------------------------------------------------------
# 8. DISPLAY MODEL
# ------------------------------------------------------------

print("\nCNN MODEL")
print("========================================")

model.summary()

# ------------------------------------------------------------
# 9. COMPILE
# ------------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ------------------------------------------------------------
# 10. TRAIN CNN
# ------------------------------------------------------------

print("\nTRAINING CNN")
print("========================================")

history = model.fit(
    train_data,
    epochs=EPOCHS
)

# ------------------------------------------------------------
# 11. TRAINING RESULTS
# ------------------------------------------------------------

training_accuracy = history.history["accuracy"][-1]
training_loss = history.history["loss"][-1]

print("\nTRAINING RESULTS")
print("========================================")

print(
    "Training Accuracy :",
    round(training_accuracy * 100, 2),
    "%"
)

print(
    "Training Loss     :",
    round(training_loss, 4)
)

# ------------------------------------------------------------
# 12. TEST MODEL
# ------------------------------------------------------------

print("\nTESTING CNN")
print("========================================")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print("\nTEST RESULTS")
print("========================================")

print(
    "Test Accuracy :",
    round(test_accuracy * 100, 2),
    "%"
)

print(
    "Test Loss     :",
    round(test_loss, 4)
)

# ------------------------------------------------------------
# 13. GET ACTUAL AND PREDICTED LABELS
# ------------------------------------------------------------

actual_labels = []
predicted_labels = []

for images, labels in test_data:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted = np.argmax(
        predictions,
        axis=1
    )

    for i in range(len(labels)):

        actual_labels.append(
            int(labels[i])
        )

        predicted_labels.append(
            int(predicted[i])
        )

actual_labels = np.array(actual_labels)
predicted_labels = np.array(predicted_labels)

# ------------------------------------------------------------
# 14. CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n")
print("====================================================")
print("              CLASSIFICATION REPORT")
print("====================================================")

all_precision = []
all_recall = []
all_f1 = []
all_support = []

for class_index in range(len(class_names)):

    TP = 0
    FP = 0
    FN = 0
    support = 0

    for i in range(len(actual_labels)):

        if actual_labels[i] == class_index:
            support = support + 1

            if predicted_labels[i] == class_index:
                TP = TP + 1

        if actual_labels[i] != class_index:

            if predicted_labels[i] == class_index:
                FP = FP + 1

        if actual_labels[i] == class_index:

            if predicted_labels[i] != class_index:
                FN = FN + 1

    # Precision

    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    # Recall

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    # F1 Score

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (
            2 * precision * recall
        ) / (precision + recall)

    all_precision.append(precision)
    all_recall.append(recall)
    all_f1.append(f1)
    all_support.append(support)

    print("\nClass:", class_names[class_index])

    print(
        "Precision :",
        round(precision, 4)
    )

    print(
        "Recall    :",
        round(recall, 4)
    )

    print(
        "F1-Score  :",
        round(f1, 4)
    )

    print(
        "Support   :",
        support
    )

# ------------------------------------------------------------
# 15. OVERALL ACCURACY
# ------------------------------------------------------------

correct = 0

for i in range(len(actual_labels)):

    if actual_labels[i] == predicted_labels[i]:
        correct = correct + 1

overall_accuracy = (
    correct / len(actual_labels)
)

print("\n====================================================")
print("OVERALL PERFORMANCE")
print("====================================================")

print(
    "Accuracy :",
    round(overall_accuracy * 100, 2),
    "%"
)

