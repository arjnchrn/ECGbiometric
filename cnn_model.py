import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np

class ECGAuthenticationCNN:
    def __init__(self, input_shape=(200, 1), num_classes=290):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.feature_extractor = None
        
    def build_model(self):
        """Build CNN model as described in the paper"""
        model = models.Sequential([
            layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=self.input_shape),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, kernel_size=5, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Conv1D(64, kernel_size=5, activation='relu'),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.2),
            
            layers.Flatten(),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=0.001):
        """Compile model with optimizer and loss"""
        if self.model is None:
            self.build_model()
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
        """Train the CNN model"""
        if self.model is None:
            self.compile_model()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        return history
    
    def create_feature_extractor(self):
        """Create feature extractor by removing FC layers"""
        if self.model is None:
            raise ValueError("Model must be trained first!")
        
        conv_layers = []
        for layer in self.model.layers:
            if isinstance(layer, (layers.Conv1D, layers.MaxPooling1D, layers.Dropout)):
                conv_layers.append(layer)
            elif isinstance(layer, layers.Flatten):
                conv_layers.append(layer)
                break
        
        feature_extractor = models.Sequential(conv_layers)
        
        self.feature_extractor = feature_extractor
        return feature_extractor
    
    def extract_features(self, X):
        """Extract high-dimensional features using convolutional layers only"""
        if self.feature_extractor is None:
            self.create_feature_extractor()
        
        features = self.feature_extractor.predict(X, verbose=0)
        return features
    
    def save_model(self, filepath='ecg_auth_model.keras'):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='ecg_auth_model.keras'):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        self.create_feature_extractor()
        print(f"Model loaded from {filepath}")
