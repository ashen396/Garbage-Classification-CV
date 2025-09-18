!pip install notebook
!pip install opencv-python
!pip install tensorflow
!pip install scikit-learn
!pip install matplotlib


import cv2

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import os
print("Current working directory:", os.getcwd())
print("Files in current dir:", os.listdir())

IMG_SIZE = 256
DATA_DIR = "./data" 

CLASSES = ["plastic", "organic", "metal"]

X, y = [], []

for idx, category in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)

        img = cv2.imread(img_path)
        if img is None:
            continue 

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        img = img / 255.0

        X.append(img)
        y.append(idx)

X = np.array(X, dtype="float32")
y = to_categorical(y, num_classes=len(CLASSES)) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset shape:", X.shape)
print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)


import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax') 
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=7,            
    batch_size=32,
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")

CLASSES = ["plastic", "organic", "metal"]

def preprocess_image(img_path, img_size=256):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    
    img = cv2.resize(img, (img_size, img_size))
    
    img = img / 255.0
    
    img = np.expand_dims(img, axis=0)
    return img

test_img_path = "./test_imgs/bottle3.jpeg"

img_ready = preprocess_image(test_img_path)

pred = model.predict(img_ready)
pred_class = np.argmax(pred, axis=1)[0]

print("Predicted class:", CLASSES[pred_class])


import matplotlib.pyplot as plt

img = cv2.imread(test_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

plt.imshow(img)
plt.title(f"Prediction: {CLASSES[pred_class]}")
plt.axis("off")
plt.show()





