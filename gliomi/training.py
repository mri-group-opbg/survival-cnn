from tensorflow.keras.preprocessing.image import ImageDataGenerator
from collections import Counter
import tensorflow as tf
import numpy as np

"""
This class balance an image dataset via augmentation.
Configuration of augmentator can be passed in constructor otherwise a simple
90° rotation, horizontal and vertical flip is provided.
"""
class DatasetBalancer:
    
    # Construct dataset balancer
    def __init__(self, augmentator=None):
        if augmentator is None:
            self.augmentator = ImageDataGenerator(
                rotation_range=90,
                horizontal_flip=True,
                vertical_flip=True
                # zoom_range=0.15,
                # width_shift_range=0.2,
                # height_shift_range=0.2,
                # shear_range=0.15,
                # fill_mode="nearest"
            )
        else:
            self.augmentator = augmentator

    """
    Given a dataset and labels return a new augmented dataset
    with a specified number of images and labels
    """
    def _fill_with_random_images(self, X, y, target_slices):

        X_new = []
        y_new = []

        # Image Generator for balancing 
        imageGenerator = self.augmentator.flow(x=X, y=y, batch_size=1)

        index = 0

        for img, label in imageGenerator:

            X_new.append(img)
            y_new.append(label)

            index = index + 1

            if index == target_slices:
                break

        return np.array(X_new), np.array(y_new)

    """
    Balance a dataset (images and labels).
    Labels must be binary.
    If target_size is not provided the subset of images with fewer labels is augmented and the other is left unvariated.
    If target_size is specified oth subset of images are generated until the target_size is reached
    """
    
    def balance(self, X, y, target_size=None):

        # Count occurrences of label 0.0 and 1.0
        count = Counter(y)

        # Bring the smaller class to the number of elements of bigger class
        if target_size is None:

            if count[0.0] < count[1.0]:
                X_zero, y_zero = self._fill_with_random_images( X[y == 0.0], y[y == 0.0], target_slices=count[1.0])
                X_one, y_one = X[y == 1.0], y[y == 1.0]
                return np.concatenate((X_zero[:,0,:,:,:], X_one), axis=0), np.concatenate((y_zero[:,0], y_one))
            else:
                X_zero, y_zero = X[y == 0.0], y[y == 0.0]
                X_one, y_one = self._fill_with_random_images( X[y == 1.0], y[y == 1.0], target_slices=count[0.0])
                return np.concatenate((X_zero, X_one[:,0,:,:,:]), axis=0), np.concatenate((y_zero, y_one[:,0]))
        else:
            X_zero, y_zero = self._fill_with_random_images(X[y == 0.0], y[y == 0.0], target_slices=target_size)
            X_one, y_one = self._fill_with_random_images(X[y == 1.0], y[y == 1.0], target_slices=target_size)
            return np.concatenate((X_zero[:,0,:,:,:], X_one[:,0,:,:,:]), axis=0), np.concatenate((y_zero[:,0], y_one[:,0]))
    
"""
Train and test a model.
This method allow to specify if dataset must be balanced
"""
class BinaryTrainer:
    
    def __init__(self, model):
        self.model = model
        
    def train_and_test(self, X_train, y_train, X_test, y_test, random_state=42, epochs=5, batch_size=16, patience=None, callbacks=[]):

        if not patience is None:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
            callbacks.append(early_stopping_callback)

        return self.model.fit(
            X_train, y_train, 
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_test, y_test), 
            shuffle=True,
            batch_size=batch_size)

    def train_and_test_generator(self, Xy_train_generator, X_test, y_test, random_state=42, epochs=5, batch_size=16, patience=None, callbacks=[]):

        if not patience is None:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience)
            callbacks.append(early_stopping_callback)

        return self.model.fit(
            Xy_train_generator, 
            epochs=epochs,
            callbacks=callbacks,
            validation_data=(X_test, y_test), 
            shuffle=True)
