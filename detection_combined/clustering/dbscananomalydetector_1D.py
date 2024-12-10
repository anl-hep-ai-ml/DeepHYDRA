#!/usr/bin/env python3
import re
import json
import logging
import multiprocessing as mp
from collections import defaultdict
from statistics import multimode

import numpy as np
import pandas as pd

# from sklearnex import patch_sklearn
# patch_sklearn(verbose=False)

# from sklearnex.cluster import DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import OrdinalEncoder

from .baseclusteringdetector import BaseClusteringDetector
from utils.anomalyclassification import AnomalyType
from utils.variables import nan_fill_value
from utils.tqdmloggingdecorator import tqdmloggingdecorator

# from sklearn2pmml.util import deep_sizeof


class HLTDBSCANAnomalyDetector(BaseClusteringDetector):

    def __init__(self,
                    node_labels: list,
                    eps: float = 3,
                    min_samples: int = 4,
                    duration_threshold: int = 4,
                    output_queue = None) -> None:
        super(HLTDBSCANAnomalyDetector, self).__init__()

        self.eps = eps
        self.min_samples = min_samples
        self.duration_threshold = duration_threshold
        self.output_queue = output_queue

        self.timesteps = []
        self.node_labels = node_labels

        #def parse_channel_name(channel_name):
            #rack_match = re.search(r'tpu-rack-(\d+)', channel_name)
            #if rack_match: rack_number = int(rack_match.group(1))
            #else: raise ValueError("Rack number not found in the channel name.")
            
            #tpu_match = re.search(r'pc-tdq-tpu-(\d+)', channel_name)
            #if tpu_match: tpu_number = int(tpu_match.group(1))
            #else: raise ValueError("TPU number not found in the channel name.")
            
            #return rack_number, tpu_number

        #self.machine_labels = [0]*len(node_labels)
        #self.rack_labels = [0]*len(node_labels)

        #for machine_index, label in enumerate(node_labels):
            #self.rack_labels[machine_index], self.machine_labels[machine_index] = parse_channel_name(label)



        self.dbscan_clustering = DBSCAN(eps=self.eps,
                                        min_samples=self.min_samples)
        

        self.anomaly_registry_general = defaultdict(int)
        self.anomaly_registry_drop_to_0 = defaultdict(int)

        self.datapoints_processed = 0

        self.memory_sizes = []


    def write_memory_size(self):
        memory_sizes = np.array(self.memory_sizes).T

        print(memory_sizes)

        memory_sizes = pd.DataFrame(memory_sizes,
                                        columns=['Memory'])

        memory_sizes.to_csv('memory_dbscan.csv')


    @tqdmloggingdecorator 
    def process(self,
                    timestep,
                    data: np.array) -> None:

        self.timesteps.append(timestep)

        subgroup_buckets = defaultdict(list)

        indices_nan = np.isnan(data)
        indices_not_nan = ~indices_nan 

        #rack_labels_filtered = np.array(self.rack_labels)[indices_not_nan]
        #machine_labels_filtered = np.array(self.machine_labels)[indices_not_nan]
        data_filtered = data[indices_not_nan]

        #cluster_predictions =\
            #self.dbscan_clustering.fit_predict(
                        #np.array(list(zip(rack_labels_filtered, data_filtered)))) #2D clustering with (rack,data), each timestamp found 3~6 clusters
        
        cluster_predictions =\
            self.dbscan_clustering.fit_predict(
                        np.array(list(data_filtered)).reshape(-1,1)) 

        num_clusters = len(set(cluster_predictions)) - (1 if -1 in cluster_predictions else 0)
    
        logging.info(f"DBSCAN found {num_clusters} clusters at timestep {timestep}.")
    
"""
        for machine_index, datapoint in enumerate(data_filtered):

            subgroup_buckets[rack_labels_filtered[machine_index]].append(
                                        [machine_labels_filtered[machine_index],
                                            datapoint,
                                            cluster_predictions[machine_index]])


        for subgroup, subgroup_bucket in subgroup_buckets.items():
            
            machine_labels_cluster, datapoints, cluster_predictions = map(list, zip(*subgroup_bucket))

            multimode_rack_bucket = multimode(cluster_predictions)
            largest_cluster_membership = multimode_rack_bucket[0]

            y_predicted = np.array([0 if cluster_prediction == largest_cluster_membership \
                                        else 1 for cluster_prediction in cluster_predictions], np.byte)

            for machine_label, datapoint, y in zip(machine_labels_cluster, datapoints, y_predicted):

                if y == 1:
                    if np.isclose(datapoint, 0):
                        self.anomaly_registry_drop_to_0[machine_label] += 1
                        self.anomaly_registry_general.pop(machine_label, None)
                    elif np.isclose(datapoint, nan_fill_value):
                        self.anomaly_registry_drop_to_0.pop(machine_label, None)
                        self.anomaly_registry_general.pop(machine_label, None)
                    else:
                        self.anomaly_registry_general[machine_label] += 1
                        self.anomaly_registry_drop_to_0.pop(machine_label, None)
                else:
                    self.anomaly_registry_drop_to_0.pop(machine_label, None)
                    self.anomaly_registry_general.pop(machine_label, None)


        cluster_anomaly_set = set()

        for machine_label, anomaly_duration in self.anomaly_registry_general.items():
            anomaly_start = self.timesteps[self.datapoints_processed - anomaly_duration + 1].strftime('%Y-%m-%d %H:%M:%S')

            if anomaly_duration == self.duration_threshold:
                 self._logger.warning(f'{machine_label}: General anomaly encountered '
                                            f'at element {self.datapoints_processed}')

            if anomaly_duration >= self.duration_threshold:
                self.detection_callback(int(machine_label),
                                            AnomalyType.ClusteringGeneral,
                                            anomaly_start,
                                            anomaly_duration)

                cluster_anomaly_set.add(machine_label//1000)

        for machine_label, anomaly_duration in self.anomaly_registry_drop_to_0.items():
            anomaly_start = self.timesteps[self.datapoints_processed - anomaly_duration + 1].strftime('%Y-%m-%d %H:%M:%S')

            if anomaly_duration == self.duration_threshold:
                 self._logger.warning(f'{machine_label}: dropped to 0 '
                                        f'at element {self.datapoints_processed}')

            if anomaly_duration >= self.duration_threshold:
                self.detection_callback(int(machine_label),
                                            AnomalyType.ClusteringDropToZero,
                                            anomaly_start,
                                            anomaly_duration)

                cluster_anomaly_set.add(machine_label//1000)

        self.datapoints_processed += 1

        if self.output_queue is not None:
            self.output_queue.put(cluster_anomaly_set)

"""
