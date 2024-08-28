from tensorflow.keras.layers import Input, Conv1D, Dropout, Conv1DTranspose
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, History
from tensorflow.keras.models import load_model, save_model

import tensorflow as tf
import os
import random
import numpy as np

class Conv_AE: 
    """
    A reconstruction convolutional autoencoder model to detect anomalies in timeseries data using reconstruction error as an anomaly score.

    Parameters
    ----------
    No parameters are required for initializing the class.

    Attributes
    ----------
    model : Sequential
        The trained convolutional autoencoder model.

    Examples
    --------
    >>> CAutoencoder = Conv_AE()
    >>> CAutoencoder.fit(train_data)
    >>> prediction = CAutoencoder.predict(test_data)
    """
    
    def __init__(self):
        self._Random(0)
        
    def _Random(self, seed_value): 

       
        os.environ['PYTHONHASHSEED'] = str(seed_value)

        
        random.seed(seed_value)

        
        np.random.seed(seed_value)

        
        tf.random.set_seed(seed_value)
        
    def _build_model(self):
        
        model = Sequential(
            [
                Input(shape=(self.shape[1], self.shape[2])),
                Conv1D(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1D(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Conv1DTranspose(
                    filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Dropout(rate=0.2),
                Conv1DTranspose(
                    filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
                ),
                Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        
        return model
    
    def fit(self, data, validation_split=0.1, epochs=40, verbose=0, shuffle=True, batch_size = 32):
        """
        Train the convolutional autoencoder model on the provided data.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for training the autoencoder model.
        """
        
        self.shape = data.shape
        self.model = self._build_model()
        
        history = History()
        
        return self.model.fit(
            data,
            data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min", verbose=0),history
            ],
        )

    def predict(self, data):
        """
        Generate predictions using the trained convolutional autoencoder model.

        Parameters
        ----------
        data : numpy.ndarray
            Input data for generating predictions.

        Returns
        -------
        numpy.ndarray
            Predicted output data.
        """
        
        return self.model.predict(data)