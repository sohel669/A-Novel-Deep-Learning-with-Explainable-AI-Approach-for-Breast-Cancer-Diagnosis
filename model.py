# model.py

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image

class BreastCancerModel:
    def __init__(self):
        # Define the same architecture as used when saving the weights
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])

        # Load the weights into the model
        self.model.load_weights(r'C:\Users\ROHIT\Downloads\BC\BC\mammography_images\fine_tuned_model_weights.h5')

    def predict(self, image_path: str) -> str:
        image = Image.open(image_path).resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        prediction = self.model.predict(image_array)[0][0]
        return 'Malignant' if prediction > 0.5 else 'Benign'
