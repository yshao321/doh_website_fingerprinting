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
    y_train = np.array(train.monitor_label)
    X_test  = combinedFeatures.transform(test)
    y_test  = np.array(test.monitor_label)
    
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
    
    # Accuracy metric
    acc = accuracy_score(y_test, yhat_test)
    print("Accuracy Score:", acc)
    
    return list(y_test), list(yhat_test)


num_classes = 10500  # Number of classes (sites)
num_samples = 20     # Number of samples for each class (site)
min_packets = 1      # Minimum of packets for each row (record)
max_packets = 50     # Maximun of packets for each row (record)

num_samples_openedworld       = 1    # Number of samples from openedworld
num_classes_monitored         = 1000 # Number of classes from monitored
num_samples_train_unmonitored = 4    # Number of training samples from unmonitored
'''
num_samples_train_monitored   = 16   # Number of training samples from monitored    (determined by k-fold: 20 * 4/5)
num_samples_train_unmonitored = 16   # Number of training samples from unmonitored  (determined by k-fold: 20 * 4/5)
num_samples_test_monitored    = 4    # Number of testing  samples from monitored    (determined by k-fold: 20 * 1/5)
num_samples_test_unmonitored  = 4    # Number of testing  samples from unmonitored  (determined by k-fold: 20 * 1/5)
'''
def classifier_train():
    # Locate dataset
    closed_data_dir = join(abspath(join(dirname("__file__"), pardir, pardir)), 'dataset', 'train')
    print(closed_data_dir)
    opened_data_dir = join(abspath(join(dirname("__file__"), pardir, pardir)), 'dataset', 'test')
    print(opened_data_dir)

    # Load dataset
    df_closed = load_data(closed_data_dir)
    print("initial closed data", df_closed.shape)
    df_opened = load_data(opened_data_dir)
    print("initial opened data", df_opened.shape)

    # Clean dataset
    df_closed = clean_df_closed(df_closed, min_packets, max_packets, num_classes, num_samples, False)
    print("cleaned closed data", df_closed.shape)
    df_opened = clean_df_opened(df_opened, min_packets, max_packets, 0, num_samples_openedworld, False)
    print("cleaned opened data", df_opened.shape)

    # Load black list classes
    black_list = load_black_list()

    # Perform k-fold cross classification
    results = []
    kf = StratifiedKFold(n_splits = 5)
    for k, (train_k, test_k) in enumerate(kf.split(df_closed, df_closed.class_label)):
        df_train_k = df_closed.iloc[train_k]
        df_test_k = df_closed.iloc[test_k]
        
        # Generate a monitor list without black list classes
        monitor_list = random.sample(list(set(df_closed.class_label.tolist())), num_classes_monitored)
        monitor_list = [x for x in monitor_list if x not in map(str, black_list)]
        
        # Training: closedworld (monitored=16 unmonitored=4)
        df_train_monitored   = df_train_k[ df_train_k["class_label"].isin(monitor_list)]
        df_train_unmonitored = df_train_k[~df_train_k["class_label"].isin(monitor_list)]
        df_train_unmonitored = select_df_by_samples(df_train_unmonitored, num_samples_train_unmonitored)
        
        df_train_monitored.insert(0, "monitor_label", "monitored")
        df_train_unmonitored.insert(0, "monitor_label", "unmonitored")
        df_train = pd.concat([df_train_monitored, df_train_unmonitored])
        
        # Testing: closedworld (monitored=4 unmonitored=4) and openedworld (monitored=1 unmonitored=1)
        df_test_monitored   = df_test_k[ df_test_k["class_label"].isin(monitor_list)]
        df_test_unmonitored = df_test_k[~df_test_k["class_label"].isin(monitor_list)]
        df_open_monitored   = df_opened[ df_opened["class_label"].isin(monitor_list)]
        df_open_unmonitored = df_opened[~df_opened["class_label"].isin(monitor_list)]
        df_test_monitored   = pd.concat([df_test_monitored, df_open_monitored])
        df_test_unmonitored = pd.concat([df_test_unmonitored, df_open_unmonitored])
        
        df_test_monitored.insert(0, "monitor_label", "monitored")
        df_test_unmonitored.insert(0, "monitor_label", "unmonitored")
        df_test = pd.concat([df_test_monitored, df_test_unmonitored])
        
        print("k-fold", k)
        start_time = time.time()
        result = classify(df_train, df_test)
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

