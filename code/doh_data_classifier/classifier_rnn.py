#!/usr/bin/env python
# coding: utf-8


import sys
import os
import time
import json
import numpy as np
import pandas as pd
import dill
import random

from os.path import join, dirname, abspath, pardir, basename
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from common.data import *


def numbers_to_sequences(samples):
    sequences = np.zeros(shape=(len(samples), max_packets, max(token_index.values()) + 1))
    for i, sample in enumerate(samples):
        for j, number in list(enumerate(sample))[:max_packets]:
            index = token_index.get(number)
            sequences[i, j, index] = 1.
    return sequences

def classify(df_train, df_valid):
    train_inputs = df_train.lengths
    train_labels = df_train.class_label
    valid_inputs = df_valid.lengths
    valid_labels = df_valid.class_label

    # Vectorize input data
    x_train = numbers_to_sequences(train_inputs)
    x_valid = numbers_to_sequences(valid_inputs)
    
    # Vectorize label data
    from keras.utils.np_utils import to_categorical
    y_train = to_categorical(train_labels)
    y_valid = to_categorical(valid_labels)

    # Define the network structure
    from keras import models
    from keras import layers
    input_dim  = x_train.shape[-1]
    output_dim = y_train.shape[-1]
    model = models.Sequential()
    model.add(layers.LSTM(256, dropout=0.2, recurrent_dropout=0, return_sequences=False, input_shape=(None, input_dim)))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(output_dim, activation='softmax'))

    # Build the network
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the network
    result = model.fit(x_train,
                       y_train,
                       epochs=100,
                       batch_size=256,
                       validation_data=(x_valid, y_valid))
    return result


num_classes = 5000   # Number of classes (sites)
num_samples = 10     # Number of samples for each class (site)
min_packets = 10     # Minimum of packets for each row (record)
max_packets = 50     # Maximun of packets for each row (record)
token_index = {}     # An index of all tokens in the data

def classifier_train():
    # Locate dataset
    data_dir = join(abspath(join(dirname("__file__"), pardir, pardir)), 'dataset', 'summer')
    print(data_dir)

    # Load dataset
    df = load_data(data_dir)
    print("initial data", df.shape)

    # Clean dataset
    df_closed = clean_df_closed(df, min_packets, max_packets, num_classes, num_samples)
    print("cleaned data", df_closed.shape)

    # Build token index
    for sample in df_closed.lengths:
        for number in sample:
            if number not in token_index:
                token_index[number] = len(token_index) + 1
    print("token index", len(token_index))

    # Perform k-fold cross classification
    train_results = []
    valid_results = []
    kf = StratifiedKFold(n_splits = 5)
    for k, (train_k, test_k) in enumerate(kf.split(df_closed, df_closed.class_label)):
        print("k-fold", k)
        start_time = time.time()
        result = classify(df_closed.iloc[train_k], df_closed.iloc[test_k])
        print("--- %s seconds ---" % (time.time() - start_time))
        train_results.append(result.history['accuracy'])
        valid_results.append(result.history['val_accuracy'])
        #break

    num_epochs = len(train_results[0])
    average_train_results = [np.mean([x[i] for x in train_results]) for i in range(num_epochs)]
    average_valid_results = [np.mean([x[i] for x in valid_results]) for i in range(num_epochs)]

    import matplotlib.pyplot as plt
    plt.plot(range(1, len(average_train_results) + 1), average_train_results)
    plt.xlabel('Epochs')
    plt.ylabel('Training ACC')
    plt.show()

    plt.clf()
    plt.plot(range(1, len(average_valid_results) + 1), average_valid_results)
    plt.xlabel('Epochs')
    plt.ylabel('Validation ACC')
    plt.show()


def classifier_serve():
    # Load pipeline
    loaded_model = dill.load(open('doh_data_classify.pickle', 'rb'))
    print("Model Loaded")

    # Load websites
    urls = get_url_list("../collection/websites.txt")

    for line in sys.stdin:
        # Locate file
        data_file = join(abspath(dirname("__file__")), line)[:-1]

        # Load file
        df_new = load_data(data_file)

        # Predict with pipeline
        pred_new = loaded_model.predict(df_new)
        pred_pro = loaded_model.predict_proba(df_new)
        pred_url = [ urls[int(index) - 1] for index in pred_new ]
        print("Prediction:", pred_url, np.max(pred_pro, axis=1))


if __name__ == '__main__':
    if (len(sys.argv) == 2):
        if (sys.argv[1] == 'train'):
            print("Training...")
            classifier_train()
            print("Training done!!!")
            exit(0)
        elif (sys.argv[1] == 'serve'):
            print("Serving...")
            classifier_serve()
            print("Serving done!!!")
            exit(0)
    print("usage: doh_data_classify.py { train | serve }")
    exit(1)


classifier_train()

