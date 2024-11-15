#!/usr/bin/env python3

import os
import math
import re
import glob
import argparse
import json
import logging
import datetime as dt
from enum import Enum
from html.parser import HTMLParser
from collections import defaultdict
from statistics import multimode

import numpy as np
from numpy.core.numeric import isclose
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.pylab as pl

from beauty import Beauty

nan_fill_value = np.finfo(np.float32).min

anomaly_duration_threshold = 8

class AtlasRunsParser(HTMLParser):

    def __init__(self):
        HTMLParser.__init__(self)

        self._runs_df = pd.DataFrame(columns = ['run', 'start', 'end', 'duration'])

        self._run_info_data_type = self.RunInfoDataType(self.RunInfoDataType.dont_care)
        self._run_info = {'run': None, 'start': None, 'end': None, 'duration': None}

    def handle_data(self, data):
        
        if self._run_info_data_type is self.RunInfoDataType.dont_care:
        
            if data == 'Run ':
                self._run_info_data_type = self.RunInfoDataType.run_number
            elif data == 'Start':
                self._run_info_data_type = self.RunInfoDataType.run_start
            elif data == 'End':
                self._run_info_data_type = self.RunInfoDataType.run_end
                
        else:
            
            if self._run_info_data_type is self.RunInfoDataType.run_number:
                
                assert(self._run_info['run'] is None)
                
                self._run_info['run'] = int(data)
                
            elif self._run_info_data_type is self.RunInfoDataType.run_start:
                
                assert(self._run_info['start'] is None)
                
                self._run_info['start'] = dt.datetime.strptime('2018 ' + data, '%Y %a %b %d, %H:%M %Z')
                
            elif self._run_info_data_type is self.RunInfoDataType.run_end:
                
                assert(self._run_info['end'] is None)
                
                self._run_info['end'] = dt.datetime.strptime('2018 ' + data, '%Y %a %b %d, %H:%M %Z')
                
                duration_dt = self._run_info['end'] - self._run_info['start']
                
                self._run_info['duration'] = duration_dt.total_seconds()
                
                self._runs_df = self._runs_df.append(self._run_info, ignore_index=True)
                
                self._run_info = {'run': None, 'start': None, 'end': None, 'duration': None}
        
            else:
                raise RuntimeError('AtlasRunsParser entered unexpected state')
                        
            self._run_info_data_type = self.RunInfoDataType.dont_care

    @property
    def runs(self):
        return self._runs_df.iloc[::-1].set_index('run')
        
    class RunInfoDataType(Enum):
        dont_care = 0
        run_number = 1
        run_start = 2
        run_end = 3


class AnomalyType(Enum):
        DTZ = 0
        DO = 1
        G = 2


class RunAnomaly():
    def __init__(self):
        self.duration = 0
        self.anomaly_types = []

    def to_json(self) -> str:

        json_dict = {'duration': self.duration}

        json_dict['types'] = [anomaly_type.name for anomaly_type in self.anomaly_types]


        return json_dict

    def update(self, duration, type):
        self.duration = duration

        if len(self.anomaly_types) != 0:
            if self.anomaly_types[-1] != type:
                self.anomaly_types.append(type)
        else:
            self.anomaly_types.append(type)


def path(directory):
    try:
        assert os.path.isdir(directory)
        return directory
    except Exception as e:
        raise argparse.ArgumentTypeError('{} is not a directory'.format(directory))


def parse_process_name(process_name):
    integers = [int(substring) for substring in re.findall(r'\d+', process_name)]

    name_components = str.split(process_name, ':')

    if len(integers) == 2:
        rack_number = integers[1]
        node_number = rack_number*1000 + 1
        clustering_label = name_components[0]
    else:
        rack_number = integers[-2]
        node_number = integers[-1]
        clustering_label = '{}_{}'.format(rack_number, name_components[0])

    return rack_number, node_number, clustering_label


def normalize_df_histogramming(process_data_pd):

    df_histogramming_mask = process_data_pd.columns.str.contains('DF_Histogramming')

    df_histogramming_data = process_data_pd.loc[:, df_histogramming_mask]

    df_histogramming_column_labels = df_histogramming_data.columns

    df_histogramming_rack_list = []

    for df_histogramming_column_label in df_histogramming_column_labels:
        df_histogramming_rack_list.append([int(substring) for substring in re.findall(
                                                                                r'\d+', df_histogramming_column_label)][-1])

    for rack in df_histogramming_rack_list:
        process_column_labels = process_data_pd.columns[process_data_pd.columns.str.contains('pc-tdq-tpu-{}'.format(rack))]

        tpu_labels = [[int(substring) for substring in re.findall(r'\d+', process_column_label)][0]\
                                                                            for process_column_label in process_column_labels]

        active_tpus = len(np.unique(tpu_labels)) - 1

        df_histogramming_label_rack = df_histogramming_column_labels.str.contains('pc-tdq-tpu-{}'.format(rack))

        df_histogramming_data.loc[:, df_histogramming_label_rack] =\
                df_histogramming_data.loc[:, df_histogramming_label_rack]/active_tpus
   
    process_data_all_pd.loc[:, df_histogramming_mask] = df_histogramming_data

    return process_data_all_pd


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Script for checking process data in the 2018 ATLAS runs'\
                                        'for anomalies using a DBSCAN clustering based anomaly detector')

    parser.add_argument('--anomaly-log-dir', nargs="?", type=path, default='../anomaly_log', help='Log file storage directory')
    parser.add_argument('--run-summary-dir', nargs="?", type=path, default='../../', help='2018 ATLAS runs summary descriptor file')

    subparsers = parser.add_subparsers(dest='mode')

    parser_partitioned = subparsers.add_parser('partitioned')
    parser_partitioned.add_argument('--run-partition-count', type=int, choices=range(1, 8), required=True)
    parser_partitioned.add_argument('--run-partition', type=int, choices=range(0, 7), required=True)

    parser_remainder = subparsers.add_parser('remainder')

    args = parser.parse_args()

    os.environ['PBEAST_SERVER_SSO_SETUP_TYPE'] = 'AutoUpdateKerberos'

    beauty_instance = Beauty()

    with open('../atlas-data-summary-runs-2018.html') as file:
        html_string = file.read()

    atlas_runs_parser = AtlasRunsParser()

    atlas_runs_parser.feed(html_string)
    atlas_runs_parser.close()

    atlas_runs_2018_df = atlas_runs_parser.runs

    if args.mode == 'partitioned':

        if args.run_partition >= args.run_partition_count:
            parser.error('Selected run partition is larger than'\
                                'the selected number of partitions')

        logging.basicConfig(filename='dbscan_anomaly_detection_'\
                                        'runs_2018_process_data_log_node_{}.log'.format(args.run_partition),
                                                                                                filemode='w',
                                                                                                level=logging.INFO,
                                                                                                format='[%(asctime)s] %(message)s',
                                                                                                datefmt='%Y-%m-%d %H:%M:%S')
        
        partition_length = math.ceil(len(atlas_runs_2018_df) / args.run_partition_count)

        partition_length_last_node = len(atlas_runs_2018_df) -\
                                        (args.run_partition_count - 1)*\
                                        partition_length

        partition_start = args.run_partition*partition_length

        partition_end = partition_start + partition_length\
                            if args.run_partition != (args.run_partition_count - 1)\
                            else partition_start + partition_length_last_node
                            
        runs_selected = atlas_runs_2018_df.iloc[partition_start:partition_end]

    elif args.mode == 'remainder':

        logging.basicConfig(filename='dbscan_anomaly_detection_runs_2018_process_data_log_remainder.log',
                                                                                            filemode='w',
                                                                                            level=logging.INFO,
                                                                                            format='[%(asctime)s] %(message)s',
                                                                                            datefmt='%Y-%m-%d %H:%M:%S')
        
        anomaly_log_file_name_list = glob.glob('{}/process_data/*.json'.format(args.anomaly_log_dir))

        run_numbers_processed = [[int(substring) for substring in re.findall(r'\d+', anomaly_log_file_name)][0]\
                                                            for anomaly_log_file_name in anomaly_log_file_name_list]

        run_numbers_all = list(atlas_runs_2018_df.index.values)

        run_numbers_not_processed = [run_number for run_number in run_numbers_all\
                                            if run_number not in run_numbers_processed]
                            
        runs_selected = atlas_runs_2018_df.loc[run_numbers_not_processed]

        # runs_selected = atlas_runs_2018_df.loc[[360309, 360161]]

        run_numbers_not_processed_string = ''.join(['\n\t{}, '.format(run_number) for run_number in run_numbers_not_processed])

        print('Processing runs {}'.format(run_numbers_not_processed))

    columns_all = None

    for run_number, run_data in runs_selected.iterrows():
        time_start = run_data['start']
        time_end = run_data['end']
        duration = run_data['duration']

        logging.info("Processing run: {}\tstart time: {}\tend time: {}\tduration: {} s".format(run_number,
                                                                                                    time_start,
                                                                                                    time_end,
                                                                                                    int(duration)))

        try:
            process_data_all_list = beauty_instance.timeseries(time_start,
                                                                time_end,
                                                                'ATLAS',
                                                                'PMGPublishedProcessData',
                                                                'resident',
                                                                'PMG.pc-tdq-tpu-.*',
                                                                regex=True,
                                                                all_publications=True)

        except RuntimeError as runtime_error:
            logging.warning('Could not read DCM rate data for run {} from PBEAST'.format(run_number))
            continue

        for count in range(1, len(process_data_all_list)):
            process_data_all_list[count] = process_data_all_list[count].alignto(process_data_all_list[0])

        process_data_all_pd = pd.concat(process_data_all_list, axis=1)

        process_data_all_pd = normalize_df_histogramming(process_data_all_pd)

        process_data_all_pd.fillna(nan_fill_value, inplace=True)

        process_data_all_list = None

        index_all = process_data_all_pd.index

        columns_all = process_data_all_pd.columns

        process_data_all_np = process_data_all_pd.to_numpy()
        
        # process_data_all_pd = pd.DataFrame(process_data_all_np,
        #                                     index=index_all,
        #                                     columns=columns_all)

        logging.info('Dataset size: {}\tchannels: {}'.format(process_data_all_pd.shape[0],
                                                                process_data_all_pd.shape[1]))

        process_data_all_pd = None

        timesteps = list(index_all)
        labels = list(columns_all.values)

        _, process_identifiers = zip(*[str.split(column, '|') for column in labels])

        rack_labels, tpu_labels, clustering_labels = zip(*[parse_process_name(process_identifier) for process_identifier in process_identifiers])

        # Temporary registries used for checking if the anomalies persist
        # for longer than the set threshold

        anomaly_registry_general = defaultdict(int)
        anomaly_registry_dropout = defaultdict(int)

        # Persistent registry used to store the timesteps at which
        # anomalies were encountered, their duration, and their type

        anomaly_registry_persistent = defaultdict(lambda: defaultdict(RunAnomaly))

        timestep_index = 0

        for timestep_index, process_data in enumerate(process_data_all_np):
            
            clustering_buckets_in = defaultdict(list)
            clustering_buckets_out = defaultdict(list)

            prediction_dict = {}

            for index, datapoint in enumerate(process_data):
                clustering_buckets_in[clustering_labels[index]].append(datapoint)

            for clustering_label, clustering_bucket in clustering_buckets_in.items():
                eps = int(np.ceil(0.5*np.median(np.where(clustering_bucket!=nan_fill_value, clustering_bucket, 0))))
                #print('{}: {}'.format(clustering_label, eps))
                # eps = 10000 
                dbscan_clustering = DBSCAN(eps=eps if eps > 0 else 1, min_samples=8)
                prediction_dict[clustering_label] = dbscan_clustering.fit_predict(np.array(clustering_bucket).reshape(-1, 1))

            for index, datapoint in enumerate(process_data):

                clustering_buckets_out[clustering_labels[index]].append([tpu_labels[index],
                                                                                    datapoint])

            # Predict process anomalies using per-rack cluster membership

            for clustering_label, clustering_bucket in clustering_buckets_out.items():
                
                cluster_predictions = prediction_dict[clustering_label]

                tpu_labels_cluster, datapoints = map(list, zip(*clustering_bucket))

                multimode_rack_bucket = multimode(cluster_predictions)
                largest_cluster_membership = multimode_rack_bucket[0]

                y_predicted = np.array([0 if cluster_prediction == largest_cluster_membership \
                                            else 1 for cluster_prediction in cluster_predictions], np.byte)

                for tpu_label, datapoint, y in zip(tpu_labels_cluster, datapoints, y_predicted):

                    process_label_full = '{}_tpu_{}'.format(clustering_label, tpu_label)

                    if y == 1:
                        if np.isclose(datapoint, nan_fill_value):
                            anomaly_registry_dropout[process_label_full] += 1
                            anomaly_registry_general.pop(process_label_full, None)
                        else:
                            anomaly_registry_general[process_label_full] += 1
                            anomaly_registry_dropout.pop(process_label_full, None)
                    else:
                        anomaly_registry_dropout.pop(process_label_full, None)
                        anomaly_registry_general.pop(process_label_full, None)

            # Add TPUs that show anomalous behavior for longer than the
            # set threshold to the persistent anomaly registry

            for tpu_label, anomaly_duration in anomaly_registry_general.items():
                anomaly_start = timesteps[timestep_index - anomaly_duration + 1].strftime('%Y-%m-%d %H:%M:%S')

                if anomaly_duration == anomaly_duration_threshold:
                    logging.info('{}: General anomaly encountered'.format(tpu_label))

                if anomaly_duration >= anomaly_duration_threshold:
                    anomaly_registry_persistent[tpu_label][anomaly_start].update(anomaly_duration,
                                                                                    AnomalyType.G)

            for tpu_label, anomaly_duration in anomaly_registry_dropout.items():
                anomaly_start = timesteps[timestep_index - anomaly_duration + 1].strftime('%Y-%m-%d %H:%M:%S')

                if anomaly_duration == anomaly_duration_threshold:
                    logging.info('{}: dropped out'.format(tpu_label))

                if anomaly_duration >= anomaly_duration_threshold:
                    anomaly_registry_persistent[tpu_label][anomaly_start].update(anomaly_duration,
                                                                                        AnomalyType.DO)

            timestep_index += 1

        if len(anomaly_registry_persistent) > 0:
            with open('{}/process_data/run_{}.json'.format(args.anomaly_log_dir, run_number), mode='w') as anomaly_log_file:
                json.dump(anomaly_registry_persistent,
                                        anomaly_log_file,
                                        indent=4,
                                        default=RunAnomaly.to_json)
