#!/usr/bin/env python
# coding: utf-8

# In[33]:


import cv2

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# In[34]:


import os
print("Current working directory:", os.getcwd())
print("Files in current dir:", os.listdir())


# In[35]:


# Parameters
IMG_SIZE = 256
DATA_DIR = "./data"  # folder structure:
# ./data/plastic/*.jpg
# ./data/organic/*.jpg
# ./data/metal/*.jpg

CLASSES = ["plastic", "organic", "metal"]

# Lists to hold data and labels
X, y = [], []

# Loop through each class folder
for idx, category in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, category)
    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            continue  # skip unreadable files

        # Step 1: Resize to fixed size (64x64)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Step 2: Normalize pixel values to [0,1]
        img = img / 255.0

        # Append to dataset
        X.append(img)
        y.append(idx)  # numeric label (0=plastic, 1=organic, 2=metal)

# Convert to numpy arrays
X = np.array(X, dtype="float32")
y = to_categorical(y, num_classes=len(CLASSES))  # one-hot encode labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Dataset shape:", X.shape)
print("Training set:", X_train.shape, y_train.shape)
print("Testing set:", X_test.shape, y_test.shape)


# In[36]:


import tensorflow as tf
from tensorflow.keras import layers, models


# In[41]:


# Define CNN model (with Input layer)
model = models.Sequential([
    layers.Input(shape=(256, 256, 3)),       # <--- recommended way

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')   # 3 output classes
])

# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# In[42]:


# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=7,            
    batch_size=32,
    verbose=1
)


# In[43]:


loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc*100:.2f}%")


# In[48]:


# Define class names (same order as Step 1)
CLASSES = ["plastic", "organic", "metal"]

def preprocess_image(img_path, img_size=256):
    # Load image with OpenCV
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    
    # Resize to CNN input size
    img = cv2.resize(img, (img_size, img_size))
    
    # Normalize to [0,1]
    img = img / 255.0
    
    # Add batch dimension (1,64,64,3)
    img = np.expand_dims(img, axis=0)
    return img


# In[56]:


# Test image
test_img_path = "./test_imgs/bottle3.jpeg"

# Preprocess image
img_ready = preprocess_image(test_img_path)

# Predict
pred = model.predict(img_ready)
pred_class = np.argmax(pred, axis=1)[0]

print("Predicted class:", CLASSES[pred_class])


# In[57]:


import matplotlib.pyplot as plt

img = cv2.imread(test_img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

plt.imshow(img)
plt.title(f"Prediction: {CLASSES[pred_class]}")
plt.axis("off")
plt.show()


# In[ ]:




