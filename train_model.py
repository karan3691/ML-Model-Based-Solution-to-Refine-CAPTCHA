import string
import numpy as np
from tensorflow.keras import layers, models
from data_preparation import load_data, encode_labels

dataset_path = "captcha_dataset"
img_size = (100, 200) 
captcha_length = 5
char_set = list(string.ascii_uppercase + string.digits)
num_chars = len(char_set)

print("Loading data...")
X, labels = load_data(dataset_path, target_size=img_size)
print("Data loaded. Preparing labels...")
Y_encoded = encode_labels(labels, char_set) 

Y_dict = {}
for i in range(captcha_length):
    Y_dict[f'char_{i}'] = Y_encoded[:, i, :]

input_shape = X.shape[1:] 
inputs = layers.Input(shape=input_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation='relu')(x)

outputs = []
for i in range(captcha_length):
    out = layers.Dense(num_chars, activation='softmax', name=f'char_{i}')(x)
    outputs.append(out)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'] * captcha_length)

model.summary()

print("Starting training...")
history = model.fit(X, Y_dict, epochs=10, batch_size=64, validation_split=0.1)
print("Training completed.")

# Save the trained model
model.save('captcha_solver_model.h5')
print("Model saved as captcha_solver_model.h5")
