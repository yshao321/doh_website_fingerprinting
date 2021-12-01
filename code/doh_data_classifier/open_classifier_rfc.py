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
import matplotlib.pyplot as plt

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


def classify(train, tests, show_curve=False):
    # Ngrams feature extractor
    combinedFeatures = FeatureUnion([('ngrams', NgramsExtractor(1, 2))])
    
    # Training and validation data
    X_train = combinedFeatures.fit_transform(train)
    y_train = np.array(train.monitor_label)
    
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
    
    for test in tests:
        X_test  = combinedFeatures.transform(test[1])
        y_test  = np.array(test[1].monitor_label)
        
        # Model evaluation
        yhat_test = rfc.predict(X_test)
        
        # Precision recall curve
        from sklearn.metrics import precision_recall_curve
        from numpy import argmax
        y_score = rfc.predict_proba(X_test)
        precision, recall, thresholds = precision_recall_curve(y_test, y_score[:,0], pos_label="monitored")
        fscore = (2 * precision * recall) / (precision + recall)
        index = argmax(fscore)
        print('Best Threshold=%f, F-Score=%.3f, Precision=%.3f, Recall=%.3f' % (thresholds[index], fscore[index], precision[index], recall[index]))
        
        plt.plot(recall, precision, label=test[0])
        plt.scatter(recall[index], precision[index], marker='o', color='black')
    
    if show_curve:
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.legend(loc="best")
        plt.title("precision vs. recall")
        plt.show()


num_classes = 10500  # Number of classes (sites)
num_samples = 20     # Number of samples for each class (site)
min_packets = 1      # Minimum of packets for each row (record)
max_packets = 50     # Maximun of packets for each row (record)

num_classes_monitored = 1000 # Number of classes to be monitored

def classifier_train():
    # Locate dataset
    closed_data_dir = join(abspath(join(dirname("__file__"), pardir, pardir)), 'dataset', 'train')
    print(closed_data_dir)

    # Load dataset
    df_closed = load_data(closed_data_dir)
    print("initial closed data", df_closed.shape)

    # Clean dataset
    df_closed = clean_df_closed(df_closed, min_packets, max_packets, num_classes, num_samples, False)
    print("cleaned closed data", df_closed.shape)

    # Generate a monitor list without black list classes
    black_list = load_black_list()
    monitor_list = random.sample(list(set(df_closed.class_label.tolist())), num_classes_monitored)
    monitor_list = [x for x in monitor_list if x not in map(str, black_list)]

    # Prepare cross-validation dataset
    df_cv = df_closed

    # Perform k-fold cross classification
    kf = StratifiedKFold(n_splits = 5)
    for k, (train_k, test_k) in enumerate(kf.split(df_cv, df_cv.class_label)):
        df_train_k = df_cv.iloc[train_k]
        df_test_k = df_cv.iloc[test_k]
        
        # Training data
        df_train_monitored   = df_train_k[ df_train_k["class_label"].isin(monitor_list)]
        df_train_unmonitored = df_train_k[~df_train_k["class_label"].isin(monitor_list)]
        
        # Training label
        df_train_monitored.insert(0, "monitor_label", "monitored")
        df_train_unmonitored.insert(0, "monitor_label", "unmonitored")
        df_train = pd.concat([df_train_monitored, df_train_unmonitored])
        
        # Testing data
        df_test_monitored   = df_test_k[ df_test_k["class_label"].isin(monitor_list)]
        df_test_unmonitored = df_test_k[~df_test_k["class_label"].isin(monitor_list)]
        
        # Testing label
        df_test_monitored.insert(0, "monitor_label", "monitored")
        df_test_unmonitored.insert(0, "monitor_label", "unmonitored")
        df_test = pd.concat([df_test_monitored, df_test_unmonitored])
        
        print("k-fold", k)
        start_time = time.time()
        classify(df_train, [(None, df_test)])
        print("--- %s seconds ---" % (time.time() - start_time))


def classifier_build():
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
    df_opened = clean_df_opened(df_opened, min_packets, max_packets, 0, 1, False)
    print("cleaned opened data", df_opened.shape)

    # Generate a monitor list without black list classes
    black_list = load_black_list()
    monitor_list = random.sample(list(set(df_closed.class_label.tolist())), num_classes_monitored)
    monitor_list = [x for x in monitor_list if x not in map(str, black_list)]

    # Training data
    df_train_monitored   = df_closed[ df_closed["class_label"].isin(monitor_list)]
    df_train_unmonitored = df_closed[~df_closed["class_label"].isin(monitor_list)]

    # Training label
    df_train_monitored.insert(0, "monitor_label", "monitored")
    df_train_unmonitored.insert(0, "monitor_label", "unmonitored")
    df_train = pd.concat([df_train_monitored, df_train_unmonitored])

    # Testing data
    df_test_monitored   = df_opened[ df_opened["class_label"].isin(monitor_list)]
    df_test_unmonitored = df_opened[~df_opened["class_label"].isin(monitor_list)]

    # Testing label
    df_test_monitored.insert(0, "monitor_label", "monitored")
    df_test_unmonitored.insert(0, "monitor_label", "unmonitored")
    df_test = pd.concat([df_test_monitored, df_test_unmonitored])

    start_time = time.time()
    classify(df_train, [('10,000', df_test[:10000]), ('20,000', df_test[:20000]), ('50,000', df_test[:50000]), ('100,000', df_test[:100000])], True)
    print("--- %s seconds ---" % (time.time() - start_time))


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

