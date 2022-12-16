import os
import pickle
import numpy as np

""" This script implements the functions for reading data.
"""

def load_data(data_dir):
    """ Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches are stored.
    
    Returns:
        x_train: An numpy array of shape [50000, 3072]. 
        (dtype=np.float32)
        y_train: An numpy array of shape [50000,]. 
        (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072]. 
        (dtype=np.float32)
        y_test: An numpy array of shape [10000,]. 
        (dtype=np.int32)
    """
    ### YOUR CODE HERE
    for data_batch in range(1, 6):
        file = data_dir + "\data_batch_" + str(data_batch)

        with open(file, 'rb') as fopen:
            raw_data = pickle.load(fopen, encoding='bytes')

        data = np.array(raw_data[b'data']).astype(np.float32)
        labels = np.array(raw_data[b'labels']).astype(np.int32)

        if data_batch == 1:
            x_train = data
            y_train = labels

        else:
            x_train = np.concatenate((x_train, data), axis=0)
            y_train = np.concatenate((y_train, labels), axis=0)

    ## Testing Set
    file_test = data_dir + "\/" + "test_batch"

    with open(file_test, 'rb') as fopen:
        raw_data = pickle.load(fopen, encoding='bytes')

    data = np.array(raw_data[b'data']).astype(np.float32)
    labels = np.array(raw_data[b'labels']).astype(np.int32)

    x_test = data
    y_test = labels
    ### YOUR CODE HERE

    return x_train, y_train, x_test, y_test

def train_vaild_split(x_train, y_train, split_index=45000):
    """ Split the original training data into a new training dataset
        and a validation dataset.
    
    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        split_index: An integer.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    return x_train_new, y_train_new, x_valid, y_valid