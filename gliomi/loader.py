import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from functools import reduce
import pickle
from keras.utils import to_categorical

class GliomiLoader:
    
    def __init__(self, dataset_path, classification_path, subjects_to_exclude=None):
        self.dataset_path = dataset_path
        self.classification_path = classification_path
        self.subjects_to_exclude = subjects_to_exclude

    """
    Compute the intersection of all keys from a given set of dataset
    """
    def subjects_in_datasets(self, datasets): 
        result = None
        for dataset in datasets:
            if result is None:
                result = list(dataset.keys())
            else:
                result = np.intersect1d(list(dataset.keys()), result)
        return result

    """
    Establish the intersection of all keys and return train and test subjects
    """
    def get_split(self, datasets, test_size=0.2, random_state=42):

        # Intersection of subjects from datasets
        subjects = self.subjects_in_datasets(datasets)

        print("Subject in intersection:", subjects.shape[0])

        indexes = list(range(subjects.shape[0]))

        train_index, test_index = train_test_split(
            indexes, 
            test_size=test_size, 
            random_state=random_state)

        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]

        return train_subjects, test_subjects

    """
    Take single slice from each subject (the first one)
    """
    def get_image_data(self, dataset, subjects):
        result = []
        for subject in subjects:
            result.append([dataset[subject][0]])
        return np.concatenate(result, axis=0)   

    """
    """
    def get_dict_data(self, dataset, subjects):
        result = []
        for subject in subjects:
            result.append(dataset[subject])
        return np.array(result)

    """
    """
    def get_labels(self, classification_path):

        y = {}

        # Load labels
        df = pd.read_csv(classification_path)
        subjects = np.array(df.iloc[:,1])
        labels = np.array(df.iloc[:,2])

        for i, subject in enumerate(subjects):
            y[subject] = labels[i]

        return y
    
    """
    Return dataset and label splice
    """
    def get_data(self):

        # Load images
        with open(self.dataset_path, "rb") as file:
            images = pickle.load(file)

        labels = self.get_labels(self.classification_path)

        if self.subjects_to_exclude:
            for exclude in self.subjects_to_exclude:
                if exclude in images:
                    del images[exclude]
                if exclude in labels:
                    del labels[exclude]

        train, test = self.get_split([images, labels])

        X_train = self.get_image_data(images, train)
        X_test = self.get_image_data(images, test)

        y_train = self.get_dict_data(labels, train)
        y_test = self.get_dict_data(labels, test)

        return X_train, y_train, X_test, y_test
    
    """
    Return dataset and label splice
    """
    def get_data_categorical(self):
        X_train, y_train, X_test, y_test = self.get_data()
        return X_train, to_categorical(y_train), X_test, to_categorical(y_test)
    