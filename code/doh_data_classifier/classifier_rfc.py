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
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from common.data import *


class NgramsExtractor:
    def __init__(self, min_ngram_len = 1, max_ngram_len = 2):
        self.positive_counter = CountVectorizer(analyzer='word',
                                                tokenizer=lambda x: x.split(),
                                                token_pattern=None,
                                                stop_words=None,
                                                ngram_range=(min_ngram_len, max_ngram_len),)
        #self.negative_counter = CountVectorizer(analyzer='word',
        #                                        tokenizer=lambda x: x.split(),
        #                                        token_pattern=None,
        #                                        stop_words=None,
        #                                        ngram_range=(min_ngram_len, max_ngram_len),)

    def fit(self, x, y = None):
        positives = x.lengths.apply(get_positives)
        #negatives = x.lengths.apply(get_negatives)
        
        self.positive_counter.fit(positives.apply(join_str))
        #self.negative_counter.fit(negatives.apply(join_str))
        
        return self

    def transform(self, data_list):
        positives = data_list.lengths.apply(get_positives)
        #negatives = data_list.lengths.apply(get_negatives)
        
        positives_str = positives.apply(join_str)
        #negatives_str = negatives.apply(join_str)
        
        positive_ngrams = self.positive_counter.transform(positives_str)
        #negative_ngrams = self.negative_counter.transform(negatives_str)
        
        #return np.concatenate((positive_ngrams.todense(), negative_ngrams.todense()), axis=1)
        return positive_ngrams.todense()

def classify(train, test):
    # Ngrams feature extractor
    combinedFeatures = FeatureUnion([('ngrams', NgramsExtractor(1, 2))])
    
    # Training and validation data
    X_train = combinedFeatures.fit_transform(train)
    y_train = np.array(train.class_label)
    X_test  = combinedFeatures.transform(test)
    y_test  = np.array(test.class_label)
    
    # Model training
    batch_mode = False
    if batch_mode:
        rfc = RandomForestClassifier(n_estimators=0, warm_start=True)
        skf = StratifiedKFold(n_splits=2)
        for _, index in skf.split(X_train, y_train):
            rfc.n_estimators += 50
            rfc.fit(X_train[index], y_train[index])
    else:
        rfc = RandomForestClassifier(n_estimators=70)
        rfc.fit(X_train, y_train)
    
    # Model evaluation
    yhat_test = rfc.predict(X_test)
    yhat_prob = np.max(rfc.predict_proba(X_test), axis=1)
    
    # Accuracy metric
    acc = accuracy_score(y_test, yhat_test)
    print("Accuracy Score:", acc)
    
    return y_test, yhat_test, yhat_prob


num_classes = 10500  # Number of classes (sites)
num_samples = 20     # Number of samples for each class (site)
min_packets = 1      # Minimum of packets for each row (record)
max_packets = 50     # Maximun of packets for each row (record)

def classifier_train():
    # Locate dataset
    data_dir = join(abspath(join(dirname("__file__"), pardir, pardir)), 'dataset', 'closed-world')
    print(data_dir)

    # Load dataset
    df = load_data(data_dir)
    print("initial data", df.shape)

    # Clean dataset
    df_cleaned = clean_df_closed(df, min_packets, max_packets, num_classes, num_samples)
    print("cleaned data", df_cleaned.shape)

    # Split dataset: train 90%, test 10%
    df_train, df_test, _, _ = train_test_split(df_cleaned, df_cleaned.class_label, test_size=0.1, stratify=df_cleaned.class_label)

    # Perform k-fold cross classification
    results = []
    kf = StratifiedKFold(n_splits = 5)
    for k, (train_k, val_k) in enumerate(kf.split(df_train, df_train.class_label)):
        print("k-fold", k)
        start_time = time.time()
        result = classify(df_train.iloc[train_k], df_train.iloc[val_k])
        print("--- %s seconds ---" % (time.time() - start_time))
        results.append(result)

    # Classification report
    reports = pd.DataFrame(columns=['k-fold', 'label', 'precision', 'recall', 'f1-score', 'support'])
    true_vectors, pred_vectors = [r[0] for r in results], [r[1] for r in results]
    for i, (y_true, y_pred) in enumerate(zip(true_vectors, pred_vectors)):
        # The precision, recall, F1 score for each class and averages in one k-fold
        output = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        report = pd.DataFrame(output).transpose()
        report = report.reset_index()
        report = report.rename(columns={'index': 'label'})
        report['k-fold'] = i
        reports = reports.append(report)

    # Statistics report
    statistics = reports.groupby('label').describe().loc['macro avg']
    print("Mean")
    print(statistics.xs('mean', level=1))
    print("Standard deviation")
    print(statistics.xs('std', level=1))

    # Test dataset accuracy
    start_time = time.time()
    classify(df_train, df_test)
    print("--- %s seconds ---" % (time.time() - start_time))

def classifier_build():
    # Locate dataset
    train_dir = join(abspath(join(dirname("__file__"), pardir, pardir)), 'dataset', 'closed-world')
    print(train_dir)
    test_dir = join(abspath(join(dirname("__file__"), pardir, pardir)), 'dataset', 'open-world')
    print(test_dir)

    # Load dataset
    df_train = load_data(train_dir)
    print("initial train data", df_train.shape)
    df_test = load_data(test_dir)
    print("initial test data", df_test.shape)

    # Clean dataset
    df_train_cleaned = clean_df_closed(df_train, min_packets, max_packets, num_classes, num_samples)
    print("cleaned train data", df_train_cleaned.shape)
    df_test_cleaned = clean_df_opened(df_test, min_packets, max_packets, 0, 1)
    print("cleaned test data", df_test_cleaned.shape)

    # Remove test labels which are not in training labels
    train_list = list(set(df_train_cleaned.class_label.tolist()))
    df_test_cleaned = df_test_cleaned[df_test_cleaned["class_label"].isin(train_list)]
    print("cleaned cleaned test data", df_test_cleaned.shape)

    start_time = time.time()
    target, prediction, probability = classify(df_train_cleaned, df_test_cleaned)
    print("--- %s seconds ---" % (time.time() - start_time))

    # Save result into CSV
    cw_result = np.column_stack((target.astype(int), prediction.astype(int), probability))
    np.savetxt("cw_result.csv", cw_result, delimiter=",", fmt="%.2f")

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
        elif (sys.argv[1] == 'build'):
            print("Building...")
            classifier_build()
            print("Building done!!!")
            exit(0)
        elif (sys.argv[1] == 'serve'):
            print("Serving...")
            classifier_serve()
            print("Serving done!!!")
            exit(0)
    print("usage: doh_data_classify.py { train | build | serve }")
    exit(1)


classifier_train()

