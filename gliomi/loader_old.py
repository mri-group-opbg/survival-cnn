from sklearn.model_selection import train_test_split
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd

import pickle

"""
Get a dataset for a given sequence with labels taken from classification CSV.
"""
def get_dataset_for_classification(dataset_path, classification_path):
    
    with open(dataset_path, "rb") as file:
        X = pickle.load(file)
        
    df = pd.read_csv(classification_path)
    subjects = np.array(df.iloc[:,1])
    labels = np.array(df.iloc[:,2])
    
    X_new = {}
    y_new = {}
    
    for i, subject in enumerate(subjects):
        if subject in X:
            X_new[subject] = X[subject]
            y_new[subject] = labels[i]
    
    return X_new, y_new

class DatasetLoader():
    
    def __init__(self, dataset_path):
        
        with open(dataset_path, "rb") as file:
            self.X, self.y = pickle.load(file)
            
        subjects = np.array(list(self.X.keys()))
        
        self.slices = np.concatenate([self.X[subject] for subject in subjects])
        self.labels = np.concatenate([np.repeat((self.y)[subject], (self.X)[subject].shape[0]) for subject in subjects])
        self.subjects = np.concatenate([np.repeat(subject, (self.X)[subject].shape[0]) for subject in subjects])
        
        # Categorical
        dictionary = np.array([[0, 1], [1, 0]])
        int_labels = self.labels.astype(int)
        self.categorical_labels = dictionary[int_labels]

    def get_subjects(self):
        return list(self.X.keys())

    def get_split(self, test_size=0.2, random_state=42):
        
        subjects = np.array(self.subjects)
        
        indexes = list(range(len(subjects)))
        
        train_index, test_index, _, _ = train_test_split(
            indexes, 
            self.labels,
            stratify=self.labels,
            test_size=test_size, 
            random_state=random_state)
        
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]
        
        X_train = self.slices[np.isin(self.subjects, train_subjects)]
        X_test = self.slices[np.isin(self.subjects, test_subjects)]
        y_train = self.labels[np.isin(self.subjects, train_subjects)]
        y_test = self.labels[np.isin(self.subjects, test_subjects)]
        
        return X_train, X_test, y_train, y_test
    
    def get_split_categorical(self, test_size=0.2, random_state=42):

        subjects = np.array(self.subjects)

        indexes = list(range(len(subjects)))

        train_index, test_index, _, _ = train_test_split(
            indexes,
            self.categorical_labels, 
            stratify=self.categorical_labels,
            test_size=test_size, 
            random_state=random_state)
        
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]
        
        X_train = self.slices[np.isin(self.subjects, train_subjects)]
        X_test = self.slices[np.isin(self.subjects, test_subjects)]
        y_train = self.categorical_labels[np.isin(self.subjects, train_subjects)]
        y_test = self.categorical_labels[np.isin(self.subjects, test_subjects)]
        
        return X_train, X_test, y_train, y_test
    
    def get_split_categorical_subject(self, test_size=0.2, random_state=42):

        subjects = np.array(self.subjects)

        indexes = list(range(len(subjects)))

        train_index, test_index, _, _ = train_test_split(
            indexes,
            self.categorical_labels, 
            stratify=self.categorical_labels,
            test_size=test_size, 
            random_state=random_state)
        
        train_subjects = subjects[train_index]
        test_subjects = subjects[test_index]
        
        X_train = self.slices[np.isin(self.subjects, train_subjects)]
        X_test = self.slices[np.isin(self.subjects, test_subjects)]
        y_train = self.categorical_labels[np.isin(self.subjects, train_subjects)]
        y_test = self.categorical_labels[np.isin(self.subjects, test_subjects)]
        
        return X_train, X_test, y_train, y_test,train_subjects
