#!/usr/bin/env python3

import re
from collections import defaultdict
import logging

import numpy as np
import pandas as pd

from .basereducer import BaseReducer
from ..utils.tqdmloggingdecorator import tqdmloggingdecorator


class MedianStdReducer(BaseReducer):

    def __init__(self, configuration_version: str) -> None:
        super(MedianStdReducer, self).__init__(configuration_version)

        self._columns_reduced = None
        self._keys_last = None

        if configuration_version == 'ECLIPSE':
            self._parse_channel_name =\
                self._parse_channel_name_eclipse
        else:
            self._parse_channel_name =\
                self._parse_channel_name_hlt_dcm


    def _parse_channel_name_hlt_dcm(self, channel_name):
        parameters = [int(substring) for substring in re.findall(r'\d+', channel_name)] # careful about overallocation configurations
        return parameters[-1]//1000


    def _parse_channel_name_eclipse(self, channel_name):
        cluster_name = channel_name.rsplit('_', maxsplit=1)[0]
        return cluster_name


    def _create_channel_names(self,
                                median_labels,
                                stdev_labels):

        median_labels = ['m_{}'.format(median_label)\
                            for median_label in median_labels]

        stdev_labels = ['std_{}'.format(stdev_label)
                            for stdev_label in stdev_labels]

        labels = np.concatenate((median_labels,
                                    stdev_labels))

        return labels
        

    @tqdmloggingdecorator
    def reduce_numpy(self,
                        machine_labels: list,
                        timestamps: list,
                        input_slice: np.array) -> pd.DataFrame:

        subgroup_numbers = [self._parse_channel_name(label) for label in machine_labels]

        # Reduce input slice

        slice_reduced_list = []

        for row_x_data in np.atleast_2d(input_slice):

            subgroup_buckets_data = defaultdict(list)

            for index, datapoint in enumerate(row_x_data):
                subgroup_buckets_data[subgroup_numbers[index]].append(datapoint)

            subgroup_median_hlt = {}
            subgroup_hlt_stdevs = {}

            for subgroup, subgroup_bucket in subgroup_buckets_data.items():
                
                subgroup_median_hlt[subgroup] = np.nanmedian(subgroup_bucket)
                subgroup_hlt_stdevs[subgroup] = np.nanstd(subgroup_bucket)

            subgroup_median_hlt = dict(sorted(subgroup_median_hlt.items()))
            subgroup_hlt_stdevs = dict(sorted(subgroup_hlt_stdevs.items()))

            if not isinstance(self._keys_last, type(None)):
                if not (subgroup_median_hlt.keys() == self._keys_last):
                    error_message_line_0 =\
                        'Subgroup bucket keys changed between slices'
                    error_message_line_1 =\
                        f'Previous keys: {self._keys_last}\t'
                    error_message_line_2 =\
                        f'Current keys: {subgroup_median_hlt.keys()}'

                    non_intersecting_keys =\
                            list(set(self._keys_last) ^\
                            set(subgroup_median_hlt.keys()))

                    error_message_line_3 =\
                        f'Keys not in both slices: {non_intersecting_keys}'

                    self._logger.error(error_message_line_0)
                    self._logger.debug(error_message_line_1)
                    self._logger.debug(error_message_line_2)
                    self._logger.debug(error_message_line_3)

                    raise RuntimeError(error_message_line_0)

                if not (subgroup_median_hlt.keys() == subgroup_hlt_stdevs.keys()):
                    error_message_line_0 =\
                        'Subgroup bucket keys not identical between '\
                        'Median and Stdev Buckets'
                    error_message_line_1 =\
                        f'Median keys: {subgroup_median_hlt.keys()}\t'
                    error_message_line_2 =\
                        f'Stdev keys: {subgroup_hlt_stdevs.keys()}'

                    non_intersecting_keys =\
                            list(set(subgroup_median_hlt.keys()) ^\
                                        set(subgroup_hlt_stdevs.keys()))

                    error_message_line_3 =\
                        f'Keys not in both: {non_intersecting_keys}'

                    self._logger.error(error_message_line_0)

                    self._logger.debug(error_message_line_1)
                    self._logger.debug(error_message_line_2)
                    self._logger.debug(error_message_line_3)

                    raise RuntimeError(error_message_line_0)

            self._keys_last = subgroup_median_hlt.keys()

            if isinstance(self._columns_reduced, type(None)):
                self._columns_reduced =\
                            self._create_channel_names(subgroup_median_hlt.keys(),
                                                            subgroup_hlt_stdevs.keys())

            subgroup_data_np = np.concatenate((np.array(list(subgroup_median_hlt.values())),
                                                np.array(list(subgroup_hlt_stdevs.values()))))

            slice_reduced_list.append(subgroup_data_np)

        slice_reduced_np = np.stack(slice_reduced_list)
        slice_reduced_np = np.nan_to_num(slice_reduced_np, nan=-1)

        # nan_amount_reduced = 100*pd.isna(slice_reduced_np.flatten()).sum()/\
        #                                                         slice_reduced_np.size

        # self._logger.debug('NaN amount reduced slice: {:.2f} %'.format(nan_amount_reduced))

        timestamps = np.atleast_1d(np.asanyarray(timestamps))

        columns_reduced_adjusted,\
                slice_reduced_np =\
                    self._adjust_reduced_data(
                                    self._columns_reduced,
                                    slice_reduced_np)

        result_slice = pd.DataFrame(slice_reduced_np,
                                            timestamps,
                                            columns_reduced_adjusted)

        return result_slice


    def reduce_pandas(self,
                        input_slice: pd.DataFrame) -> pd.DataFrame:

        machine_labels = list((input_slice).columns.values)
        timestamps = list(input_slice.index)

        input_slice_np = input_slice.to_numpy()

        return self.reduce_numpy(machine_labels,
                                    timestamps,
                                    input_slice_np)


    # Function to extract rack number from a column name
    def extract_rack_number(self, column_name):
        match = re.search(r'tpu-rack-(\d+)', column_name)
        if match:
            return int(match.group(1))
        else:
            return np.nan  # Use NaN for columns without a rack number

    @tqdmloggingdecorator
    def reduce_bulk_offline(self,
                        hlt_data_pd: pd.DataFrame) -> pd.DataFrame:


        # Apply the function to all column names
        rack_numbers = [self.extract_rack_number(col) for col in hlt_data_pd.columns]

        # Create a DataFrame mapping columns to rack numbers
        columns_df = pd.DataFrame({
            'column_name': hlt_data_pd.columns,
            'rack_number': rack_numbers
        })
        print(columns_df)

        #subgroup_labels_expected_hlt_dcm_2023 =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95]

        # Inheritance from BaseReducer
        subgroup_labels_expected_hlt_dcm_2023 = self._subgroup_numbers_expected

        # Convert rack numbers to integers and drop NaNs
        columns_df = columns_df.dropna(subset=['rack_number'])
        columns_df['rack_number'] = columns_df['rack_number'].astype(int)

        # Filter columns for expected racks
        expected_racks = subgroup_labels_expected_hlt_dcm_2023
        columns_df = columns_df[columns_df['rack_number'].isin(expected_racks)]
        print(columns_df)

        rack_to_columns = defaultdict(list)
        for _, row in columns_df.iterrows():
            rack_to_columns[row['rack_number']].append(row['column_name'])

        print(rack_to_columns)
        timestamps = hlt_data_pd.index

        # Initialize DataFrames with NaNs
        medians_df = pd.DataFrame(index=timestamps, columns=expected_racks)
        stds_df = pd.DataFrame(index=timestamps, columns=expected_racks)

        for rack in expected_racks:
            columns = rack_to_columns.get(rack, [])
            if columns:
                data = hlt_data_pd[columns]
                # Compute median and std across the columns for each timestamp
                medians_df[rack] = data.median(axis=1)
                stds_df[rack] = data.std(axis=1)
            else:
                # If no data for the rack, fill with dummy values or leave as NaN
                medians_df[rack] = 0  
                stds_df[rack] = 0  
        print(medians_df.iloc[500:501])
        print(stds_df.iloc[500:501])


        # Rename columns to indicate median and std
        medians_df = medians_df.add_prefix('m_')
        stds_df = stds_df.add_prefix('std_')

        # Concatenate along the columns
        reduced_data_df = pd.concat([medians_df, stds_df], axis=1)

        print(reduced_data_df.iloc[500:501])

        return reduced_data_df



