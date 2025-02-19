import os
import string
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(dataset_path, target_size=(100, 200)):
    images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith('.png'):
            img = load_img(os.path.join(dataset_path, filename), target_size=target_size)
            img_array = img_to_array(img) / 255.0  
            images.append(img_array)
            label = filename.split('_')[0]
            labels.append(label)
    return np.array(images), labels

def encode_labels(labels, char_set):
    captcha_length = len(labels[0])
    encoded = []
    for label in labels:
        onehot_chars = []
        for char in label:
            one_hot = [0] * len(char_set)
            index = char_set.index(char)
            one_hot[index] = 1
            onehot_chars.append(one_hot)
        encoded.append(onehot_chars)
    return np.array(encoded)

if __name__ == "__main__":
    dataset_path = "captcha_dataset"
    X, labels = load_data(dataset_path)
    char_set = list(string.ascii_uppercase + string.digits)
    Y = encode_labels(labels, char_set)
    print("Loaded data:")
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
