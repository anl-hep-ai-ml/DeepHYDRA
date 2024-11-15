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

from tqdm import tqdm
from tqdm.contrib import tzip
from tqdm.contrib.logging import logging_redirect_tqdm

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



class AtlasRunsParser(HTMLParser):

    def __init__(self, year: int):
        HTMLParser.__init__(self)

        self._year = year

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
                
                self._run_info['start'] = dt.datetime.strptime(f'{self._year} ' + data, '%Y %a %b %d, %H:%M %Z')
                
            elif self._run_info_data_type is self.RunInfoDataType.run_end:
                
                assert(self._run_info['end'] is None)
                
                self._run_info['end'] = dt.datetime.strptime(f'{self._year} ' + data, '%Y %a %b %d, %H:%M %Z')
                
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--anomaly-log-dir', nargs="?", type=path, default='./results', help='Log file storage directory')
    parser.add_argument('--run-summary-file', nargs="?", type=str)
    parser.add_argument('--year', type=int, default=2023)
    parser.add_argument('--duration-threshold', type=int, default=4)

    subparsers = parser.add_subparsers(dest='mode')

    parser_partitioned = subparsers.add_parser('partitioned')
    parser_partitioned.add_argument('--run-partition-count', type=int, choices=range(1, 8), required=True)
    parser_partitioned.add_argument('--run-partition', type=int, choices=range(0, 7), required=True)

    parser_remainder = subparsers.add_parser('remainder')

    args = parser.parse_args()

    os.environ['PBEAST_SERVER_SSO_SETUP_TYPE'] = 'AutoUpdateKerberos'

    beauty_instance = Beauty('https://vm-atlas-tdaq-cc.cern.ch/tbed/pbeast/api/')

    with open(args.run_summary_file) as file:
        html_string = file.read()

    atlas_runs_parser = AtlasRunsParser(args.year)

    atlas_runs_parser.feed(html_string)
    atlas_runs_parser.close()

    atlas_runs_df = atlas_runs_parser.runs

    if args.mode == 'partitioned':

        if args.run_partition >= args.run_partition_count:
            parser.error('Selected run partition is larger than'\
                                'the selected number of partitions')

        logging.basicConfig(filename='dbscan_anomaly_detection_'\
                                        f'runs_{args.year}_log_node_{args.run_partition}.log',
                                                                                    filemode='w',
                                                                                    level=logging.INFO,
                                                                                    format='[%(asctime)s] %(message)s',
                                                                                    datefmt='%Y-%m-%d %H:%M:%S')
        
        partition_length = math.ceil(len(atlas_runs_df) / args.run_partition_count)

        partition_length_last_node = len(atlas_runs_df) -\
                                        (args.run_partition_count - 1)*\
                                        partition_length

        partition_start = args.run_partition*partition_length

        partition_end = partition_start + partition_length\
                            if args.run_partition != (args.run_partition_count - 1)\
                            else partition_start + partition_length_last_node
                            
        atlas_runs_df = atlas_runs_df.iloc[partition_start:partition_end]

    elif args.mode == 'remainder':

        logging.basicConfig(filename=f'dbscan_anomaly_detection_runs_{args.year}_log_remainder.log',
                                                                            filemode='w',
                                                                            level=logging.INFO,
                                                                            format='[%(asctime)s] %(message)s',
                                                                            datefmt='%Y-%m-%d %H:%M:%S')
        
        anomaly_log_file_name_list = glob.glob('{}/*.json'.format(args.anomaly_log_dir))

        run_numbers_processed = [[int(substring) for substring in re.findall(r'\d+', anomaly_log_file_name)][0]\
                                                            for anomaly_log_file_name in anomaly_log_file_name_list]

        run_numbers_all = list(atlas_runs_df.index.values)

        run_numbers_not_processed = [run_number for run_number in run_numbers_all\
                                            if run_number not in run_numbers_processed]
                            
        atlas_runs_df = atlas_runs_df.loc[run_numbers_not_processed]

        run_numbers_not_processed_string = ''.join(['\n\t{}, '.format(run_number) for run_number in run_numbers_not_processed])

        print('Processing runs {}'.format(run_numbers_not_processed))

    columns_all = None

    with logging_redirect_tqdm():
        for run_number, run_data in tqdm(atlas_runs_df.iterrows(), desc=f"Executing Run partition: {args.run_partition}"):
            time_start = run_data['start']
            time_end = run_data['end']
            duration = run_data['duration']

            logging.info("Processing run: {}\tstart time: {}\tend time: {}\tduration: {} s".format(run_number,
                                                                                                        time_start,
                                                                                                        time_end,
                                                                                                        int(duration)))


            try:
                dcm_rates_all_list = beauty_instance.timeseries(time_start,
                                                                    time_end,
                                                                    'ATLAS',
                                                                    'DCM',
                                                                    'L1Rate',
                                                                    'DF_IS:.*.DCM.*.info',
                                                                    regex=True,
                                                                    all_publications=True)

            except RuntimeError as runtime_error:
                logging.warning('Could not read DCM rate data for run {} from PBEAST'.format(run_number))
                continue

            for count in range(1, len(dcm_rates_all_list)):
                dcm_rates_all_list[count] = dcm_rates_all_list[count].alignto(dcm_rates_all_list[0])

            dcm_rates_all_pd = pd.concat(dcm_rates_all_list, axis=1)

            dcm_rates_all_list = None

            index_all = dcm_rates_all_pd.index

            columns_all = dcm_rates_all_pd.columns

            dcm_rates_all_np = dcm_rates_all_pd.to_numpy()
            
            dcm_rates_all_pd = pd.DataFrame(dcm_rates_all_np,
                                                index=index_all,
                                                columns=columns_all)

            dcm_rates_all_pd.fillna(nan_fill_value, inplace=True)

            dcm_rates_all_np = dcm_rates_all_pd.to_numpy()

            logging.info('Dataset size: {}\tchannels: {}'.format(dcm_rates_all_np.shape[0],
                                                                    dcm_rates_all_np.shape[1]))

            timesteps = list(dcm_rates_all_pd.index)
            node_labels = list((dcm_rates_all_pd).columns.values)

            dcm_rates_all_pd = None

            def parse_channel_name(channel_name):
                parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)]
                return parameters[1], parameters[4]

            tpu_labels = [0]*len(node_labels)
            rack_labels = [0]*len(node_labels)

            for tpu_index, label in enumerate(node_labels):
                rack_labels[tpu_index], tpu_labels[tpu_index] = parse_channel_name(label)

            dbscan_clustering = DBSCAN(eps=3, min_samples=4)

            # Temporary registries used for checking if the anomalies persist
            # for longer than the set threshold

            anomaly_registry_general = defaultdict(int)
            anomaly_registry_drop_to_0 = defaultdict(int)
            anomaly_registry_dropout = defaultdict(int)

            # Persistent registry used to store the timesteps at which
            # anomalies were encountered, their duration, and their type

            anomaly_registry_persistent = defaultdict(lambda: defaultdict(RunAnomaly))

            timestep_index = 0

            for timestep_index, dcm_rates in enumerate(dcm_rates_all_np):
                rack_buckets = defaultdict(list)

                cluster_predictions = dbscan_clustering.fit_predict(np.array(list(zip(rack_labels, dcm_rates))))

                for tpu_index, datapoint in enumerate(dcm_rates):

                    rack_buckets[rack_labels[tpu_index]].append([tpu_labels[tpu_index],
                                                                                datapoint,
                                                                                cluster_predictions[tpu_index]])

                # Predict TPU anomalies using per-rack cluster membership

                for rack, rack_bucket in rack_buckets.items():
                    
                    tpu_labels_cluster, datapoints, cluster_predictions = map(list, zip(*rack_bucket))

                    multimode_rack_bucket = multimode(cluster_predictions)
                    largest_cluster_membership = multimode_rack_bucket[0]

                    y_predicted = np.array([0 if cluster_prediction == largest_cluster_membership \
                                                else 1 for cluster_prediction in cluster_predictions], np.byte)

                    for tpu_label, datapoint, y in zip(tpu_labels_cluster, datapoints, y_predicted):

                        if y == 1:
                            if np.isclose(datapoint, 0):
                                anomaly_registry_drop_to_0[tpu_label] += 1
                                anomaly_registry_dropout.pop(tpu_label, None)
                                anomaly_registry_general.pop(tpu_label, None)
                            elif np.isclose(datapoint, nan_fill_value):
                                anomaly_registry_dropout[tpu_label] += 1
                                anomaly_registry_drop_to_0.pop(tpu_label, None)
                                anomaly_registry_general.pop(tpu_label, None)
                            else:
                                anomaly_registry_general[tpu_label] += 1
                                anomaly_registry_drop_to_0.pop(tpu_label, None)
                                anomaly_registry_dropout.pop(tpu_label, None)
                        else:
                            anomaly_registry_drop_to_0.pop(tpu_label, None)
                            anomaly_registry_dropout.pop(tpu_label, None)
                            anomaly_registry_general.pop(tpu_label, None)

                # Add TPUs that show anomalous behavior for longer than the
                # set threshold to the persistent anomaly registry

                for tpu_label, anomaly_duration in anomaly_registry_general.items():
                    anomaly_start = timesteps[timestep_index - anomaly_duration + 1].strftime('%Y-%m-%d %H:%M:%S')

                    if anomaly_duration == args.duration_threshold:
                        logging.info('{}: General anomaly encountered'.format(tpu_label))

                    if anomaly_duration >= args.duration_threshold:
                        anomaly_registry_persistent[tpu_label][anomaly_start].update(anomaly_duration,
                                                                                        AnomalyType.G)

                for tpu_label, anomaly_duration in anomaly_registry_drop_to_0.items():
                    anomaly_start = timesteps[timestep_index - anomaly_duration + 1].strftime('%Y-%m-%d %H:%M:%S')

                    if anomaly_duration == args.duration_threshold:
                        logging.info('{}: dropped to 0'.format(tpu_label))

                    if anomaly_duration >= args.duration_threshold:
                        anomaly_registry_persistent[tpu_label][anomaly_start].update(anomaly_duration,
                                                                                            AnomalyType.DTZ)

                for tpu_label, anomaly_duration in anomaly_registry_dropout.items():
                    anomaly_start = timesteps[timestep_index - anomaly_duration + 1].strftime('%Y-%m-%d %H:%M:%S')

                    if anomaly_duration == args.duration_threshold:
                        logging.info('{}: dropped out'.format(tpu_label))

                    if anomaly_duration >= args.duration_threshold:
                        anomaly_registry_persistent[tpu_label][anomaly_start].update(anomaly_duration,
                                                                                            AnomalyType.DO)

                timestep_index += 1

            if len(anomaly_registry_persistent) > 0:
                with open('{}/run_{}.json'.format(args.anomaly_log_dir, run_number), mode='w') as anomaly_log_file:
                    json.dump(anomaly_registry_persistent,
                                            anomaly_log_file,
                                            indent=4,
                                            default=RunAnomaly.to_json)
