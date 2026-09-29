import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Dataset path
path = input("Enter dataset path: ")

# Load data
train = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="training",
    seed=123, image_size=(32, 32), batch_size=32
)

test = tf.keras.utils.image_dataset_from_directory(
    path, validation_split=0.2, subset="validation",
    seed=123, image_size=(32, 32), batch_size=32
)

classes = train.class_names

# Normalize
train = train.map(lambda x, y: (x / 255.0, y))
test = test.map(lambda x, y: (x / 255.0, y))

# LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input((32, 32, 3)),

    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(2),

    tf.keras.layers.Reshape((15, 240)),
    tf.keras.layers.LSTM(32),

    tf.keras.layers.Dense(5, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
model.fit(train, epochs=3)

# Test
loss, accuracy = model.evaluate(test)
print("Test Accuracy:", accuracy)

# Prediction
for images, labels in test.take(1):

    prediction = model.predict(images[:1], verbose=0)
    p = np.argmax(prediction[0])

    print("Actual   :", classes[labels[0]])
    print("Predicted:", classes[p])

    plt.imshow(images[0])
    plt.title(
        "Actual: " + classes[labels[0]] +
        " | Predicted: " + classes[p]
    )
    plt.axis("off")
    plt.show()

    break

print("LSTM completed successfully!")
