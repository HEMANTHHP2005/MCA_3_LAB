import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Paths and parameters
train_path = input("Enter TRAIN folder path: ")
test_path = input("Enter TEST folder path: ")

IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 3

# Get PNG files
def get_files(path):
    return [os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(".png")]

train_files = get_files(train_path)[:5000]
test_files = get_files(test_path)[:1000]


print("Train images:", len(train_files))
print("Test images :", len(test_files))

# Load image
def prepare(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, img

# Dataset
train_data = (tf.data.Dataset.from_tensor_slices(train_files)
              .map(prepare)
              .shuffle(500)
              .batch(BATCH_SIZE))

test_data = (tf.data.Dataset.from_tensor_slices(test_files)
             .map(prepare)
             .batch(BATCH_SIZE))

# Autoencoder
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(2, padding="same"),
    tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D(2, padding="same"),
    tf.keras.layers.Conv2DTranspose(32, 3, strides=2,
                                     activation="relu", padding="same"),
    tf.keras.layers.Conv2DTranspose(16, 3, strides=2,
                                     activation="relu", padding="same"),
    tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same")
])

# Compile and train
model.compile(optimizer="adam", loss="mse")

model.fit(train_data, epochs=EPOCHS, validation_data=test_data)

# Test
loss = model.evaluate(test_data)
print("Test Loss:", loss)

# Display results
for images, _ in test_data.take(1):
    decoded = model.predict(images[:5], verbose=0)

    plt.figure(figsize=(10, 4))

    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap="gray")
        plt.axis("off")
        plt.title("Original")

        plt.subplot(2, 5, i + 6)
        plt.imshow(decoded[i].squeeze(), cmap="gray")
        plt.axis("off")
        plt.title("Decoded")

    plt.show()

print("Autoencoder completed!")
