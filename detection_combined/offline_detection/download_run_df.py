#!/usr/bin/env python3
import argparse
import sys
import os
import json
import numpy as np
import pandas as pd
sys.path.append('../')
from utils.offlinepbeastdataloader import OfflinePBeastDataLoader


def _remove_timestamp_jumps(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Replace large timestamp jumps resulting from
    # consecutive datapoints coming from different
    # runs with a delta of 5 s, which is the average
    # update frequency of L1 rate data
    delta = index[1:] - index[:-1]

    index = pd.Series(index)

    for i in range(1, len(index)):
        if delta[i - 1] >= pd.Timedelta(10, unit='s'):

            index[i:] = index[i:] - delta[i - 1] +\
                            pd.Timedelta(5, unit='s')

    index = pd.DatetimeIndex(index)
    assert index.is_monotonic_increasing
    return index


#def load_anomaly_machines(json_file: str):
#    with open(json_file, 'r') as file:
#        anomaly_data = json.load(file)
#    return anomaly_data


def load_anomaly_machines(json_dir: str, run_number):
    anomaly_data = {}
    for file_name in os.listdir(json_dir):
        if file_name.endswith('.json'):
            if str(run_number) in file_name:
                with open(os.path.join(json_dir, file_name), 'r') as file:
                    data = json.load(file)
                    anomaly_data.update(data)
    return anomaly_data


def filter_columns(anomaly_data, df):
    
    relevant_machines = set(machine.zfill(5) for machine in anomaly_data.keys())
    rack_prefixes = set(machine[:2] for machine in relevant_machines)
    additional_columns = set() 

    for rack_prefix in rack_prefixes:
        # Find a column that matches the rack prefix and is not already in the relevant_machines
        for col in df.columns:
            if f"tpu-rack-{rack_prefix}" in col and col not in relevant_machines:
                additional_columns.add(col)
                break  # Only adding one column per rack prefix

    # Combine relevant machines and additional columns
    relevant_columns = list(relevant_machines) + list(additional_columns)
    filtered_columns = [col for col in df.columns if any(str(machine_id) in col for machine_id in relevant_columns)]
    
    return df[filtered_columns]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Downloading dataframe for a given run')
    parser.add_argument('--run_numbers', type=str)
    parser.add_argument('--anomaly_json_dir', type=str, required=True)
    args = parser.parse_args()

    #offline_pbeast_data_loader = OfflinePBeastDataLoader('../../../atlas-data-summary-runs-2018.html')
    offline_pbeast_data_loader = OfflinePBeastDataLoader('../../datasets/atlas-data-summary-runs-2023.html')
    run_numbers = offline_pbeast_data_loader.get_run_numbers()
    
    print(run_numbers)

    missing_run_numbers = [int(run) for run in args.run_numbers.split(',')]
    print(missing_run_numbers)

    for run_number in missing_run_numbers:
        if run_number in run_numbers:
            print(f'Starting data downloading for run {run_number}')
            hlt_data_pd = offline_pbeast_data_loader[run_number]
            hlt_data_pd.index = _remove_timestamp_jumps(pd.DatetimeIndex(hlt_data_pd.index))
            #hlt_data_pd.to_csv('hlt_data_pd_'+str(run_number)+'.csv', index=True)
            # Load anomaly data and filter columns
            if args.anomaly_json_dir != "None":
                anomaly_data = load_anomaly_machines(args.anomaly_json_dir, run_number)
                filtered_df = filter_columns(anomaly_data, hlt_data_pd)
            
                filtered_df.to_csv(f'hlt_data_pd_{run_number}.csv', index=True)
            else:
                hlt_data_pd.to_csv(f'hlt_data_pd_{run_number}.csv', index=True)
        else:
            print(f'Error: The specified run number {str(run_number)} was not found. Check the atlas data summary runs list.')
