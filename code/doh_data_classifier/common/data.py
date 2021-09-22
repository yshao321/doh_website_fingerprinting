#!/usr/bin/env python
# coding: utf-8


import os
import json
import numpy as np
import pandas as pd

from functools import partial
from os.path import join, dirname, abspath, pardir, basename

def select_df_by_samples(df, num_samples):
    df2 = df.copy()
    df2['selected'] = False

    def select_elements_if_enough(elements, num):
        if len(elements) >= num:
            elements[:num] = True
        return elements

    groups = df2.groupby('class_label')
    p_select = partial(select_elements_if_enough, num = num_samples)
    df2['selected'] = groups.selected.transform(p_select)

    df_selected = df[df2.selected]
    return df_selected

def select_df_by_packets(df, min_packets, max_packets):
    df.lengths = [ value[value > 0]    for index, value in df.lengths.items() ]
    df.lengths = [ value[:max_packets] for index, value in df.lengths.items() ]
    
    selected = [ value.size >= min_packets for index, value in df.lengths.items() ]
    df_selected = df[selected]
    return df_selected

def select_df_by_max_classes(df, max_classes):
    selected = df.class_label.isin(map(str, range(max_classes + 1)))
    df_selected = df[selected]
    return df_selected

def clean_df_closed(df, min_packets, max_packets, max_classes, num_samples):
    df = select_df_by_packets(df, min_packets, max_packets)
    df = select_df_by_samples(df, num_samples)
    df = select_df_by_max_classes(df, max_classes)
    
    df = df.sort_values('class_label')
    df.index = range(len(df.index))

    return df

def select_df_by_min_classes(df, min_classes):
    selected = ~df.class_label.isin(map(str, range(min_classes + 1)))
    df_selected = df[selected]
    return df_selected

def clean_df_opened(df, min_packets, max_packets, min_classes, num_samples):
    df = select_df_by_packets(df, min_packets, max_packets)
    df = select_df_by_samples(df, num_samples)
    df = select_df_by_min_classes(df, min_classes)
    
    df = df.sort_values('class_label')
    df.index = range(len(df.index))

    return df

def parse_file(fpath):
    with open(fpath) as f:
        data_dict = json.loads(f.read())
        try:
            for keys, values in data_dict.items():
                site_id = keys
                site_lengths = np.array(values['lengths'])
                yield site_id, site_lengths
        except Exception as e:
            print ("ERROR:", fpath, e)

def select_df_by_black_files(df, dpath):
    black_list = []
    for root, _, files in os.walk(dpath):
        for fname in files:
            if not fname.endswith('.blk'):
                continue
            black_file = os.path.join(root, fname)
            df_black = pd.read_csv(black_file, header=None)
            black_list.extend(df_black.iloc[:,0].tolist())
    
    selected = ~df.class_label.isin(map(str, black_list))
    df_selected = df[selected]
    return df_selected

def load_data(path):
    selected_files = []
    pickle_file = None

    if os.path.isfile(path):
        fpath = path
        selected_files.append(fpath)
    else:
        pickle_file = join(path, '%s.pickle' % os.path.basename(path))
        if os.path.isfile(pickle_file):
            print("read data from pickle")
            df = pd.read_pickle(pickle_file)
            df = select_df_by_black_files(df, path)
            return df
        
        dpath = path
        for root, _, files in os.walk(dpath):
            for fname in files:
                if not fname.endswith('.json'):
                    continue
                fpath = os.path.join(root, fname)
                selected_files.append(fpath)

    df = pd.DataFrame()
    for fpath in selected_files:
        row = {}
        for i, (site_id, site_lengths) in enumerate(parse_file(fpath)):
            row['fname'] = os.path.basename(fpath)
            row['class_label'] = site_id
            row['lengths'] = site_lengths
            df = df.append(row, ignore_index=True)
        print (i + 1, fpath)

    if pickle_file is not None:
        print("save data into pickle")
        df.to_pickle(pickle_file)

    df = select_df_by_black_files(df, path)
    return df

def join_str(lengths):
    return ' '.join(map(str, lengths))

def get_positives(len_seq):
    return len_seq[len_seq > 0]

def get_negatives(len_seq):
    return len_seq[len_seq < 0]

def get_url_list(url_list):
    urls = []
    with open(url_list) as f:
        lines = f.readlines()
        urls = [x.strip() for x in lines]
    return urls

