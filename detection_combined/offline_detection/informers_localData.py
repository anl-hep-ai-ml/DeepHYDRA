#!/usr/bin/env python3

import argparse
import sys
import os
import datetime as dt
import json
import logging
import time 
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
from tqdm.contrib.logging import logging_redirect_tqdm

sys.path.append('../../')

from clustering.dbscananomalydetector import HLTDBSCANAnomalyDetector
from reduction.medianstdreducer import MedianStdReducer
from transformer_based_detection.informers.informerrunner import InformerRunner
from utils.anomalyregistry import JSONAnomalyRegistry
from utils.reduceddatabuffer import ReducedDataBuffer
from utils.exceptions import NonCriticalPredictionException
from utils.consolesingleton import ConsoleSingleton

start_time = time.time()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='T-DBSCAN/Informer Offline HLT Anomaly Detection')

    parser.add_argument('--model', type=str, choices=['Informer-MSE', 'Informer-SMSE'])
    parser.add_argument('--checkpoint-dir', type=str, default='../../../transformer_based_detection')
    parser.add_argument('--data-dir', type=str, default='../../../datasets/hlt/')
    parser.add_argument('--inp-data-name', type=str, default='test_set_dcm_rates_2023.csv')
    
    parser.add_argument('--output-dir', type=str, default='./results/')
    parser.add_argument('--log-level', type=str, default='info')
    parser.add_argument('--log-dir', type=str, default='./log/')
    
    parser.add_argument('--dbscan-eps', type=float, default=0.6)
    parser.add_argument('--dbscan-min-samples', type=int, default=4)
    parser.add_argument('--dbscan-duration-threshold', type=int, default=4)

    parser.add_argument('--variant', type=str, choices=['2018', '2022', '2023'], default='2018')
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()

    time_now_string = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    log_model_name = args.model.lower().replace('-', '_')

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    log_filename = f'{args.log_dir}/strada_{log_model_name}'\
                            f'benchmark_log_{time_now_string}.log'

    logging_format = '[%(asctime)s] %(levelname)s: %(name)s: %(message)s'

    logger = logging.getLogger(__name__)

    logging.getLogger().addFilter(lambda record: 'running accelerated version on CPU' not in record.msg)

    logging.basicConfig(filename=log_filename,
                                    filemode='w',
                                    level=args.log_level.upper(),
                                    format=logging_format,
                                    datefmt='%Y-%m-%d %H:%M:%S')

    inp_data_name = args.inp_data_name

    logger.info(f'Starting data loading for {args.inp_data_name}')

    hlt_data_pd = pd.read_csv(f'{args.data_dir}/{args.inp_data_name}', index_col=0, parse_dates=True)


    rack_config = '2018' if args.variant in ['2018', '2022'] else '2023'

    median_std_reducer = MedianStdReducer(rack_config)
    
    informer_runner = InformerRunner(args.checkpoint_dir, device='cpu')

    tpu_labels = list(hlt_data_pd.columns.values)

    logger.info('Instantiating T-DBSCAN detector with parameters'
                                    f'DBSCAN ε: {args.dbscan_eps} '
                                    f'min_samples: {args.dbscan_min_samples} '
                                    f'duration threshold: {args.dbscan_duration_threshold}')

    dbscan_anomaly_detector =\
        HLTDBSCANAnomalyDetector(tpu_labels,
                                    args.dbscan_eps,
                                    args.dbscan_min_samples,
                                    args.dbscan_duration_threshold)

    logger.info('Successfully instantiated T-DBSCAN detector')
    logger.info(f'Instantiating model {args.model}')

    if args.model == 'Informer-SMSE':
        reduced_data_buffer = ReducedDataBuffer(size=65)

    else:
        reduced_data_buffer = ReducedDataBuffer(size=17)

    logger.info(f'Successfully instantiated model {args.model}')

    reduced_data_buffer.set_buffer_filled_callback(informer_runner.detect)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    json_anomaly_registry = JSONAnomalyRegistry(args.output_dir)

    dbscan_anomaly_detector.register_detection_callback(
                    json_anomaly_registry.clustering_detection)
    informer_runner.register_detection_callback(
                    json_anomaly_registry.transformer_detection)

    timestamps = list(hlt_data_pd.index)
    hlt_data_np = hlt_data_pd.to_numpy()

    logger.info(f'Starting combined detection on data of run {args.inp_data_name}')

    with logging_redirect_tqdm():
        for count, (timestamp, data) in enumerate(tzip(timestamps, hlt_data_np)):
            try:
                #if count==0: # Process only the first timestamp
                    dbscan_anomaly_detector.process(timestamp, data)

                    #output_slice =\
                        #median_std_reducer.reduce_numpy(tpu_labels,
                                                       #timestamp,
                                                       #data)
                    #reduced_data_buffer.push(output_slice)
                #else:
                    #break  # Exit loop after processing the first timestamp

            except NonCriticalPredictionException:
                break

    logger.info(f'Processing of data of {args.inp_data_name} finished')

    log_file_name = f'log_{args.inp_data_name}'

    json_anomaly_registry.write_log_file(log_file_name)

    logger.info(f'Exported results for {args.inp_data_name} '
                        f'to file {log_file_name}.json')

end_time = time.time()
total_time = end_time - start_time
print(f"Total running time: {total_time:.2f} seconds")
