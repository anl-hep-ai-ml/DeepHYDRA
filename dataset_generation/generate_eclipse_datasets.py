#!/usr/bin/env python3
import math
import re
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

max_val = 100

image_width = 1920
image_height = 1080

plot_window_size = 100

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255,255,255)
thickness = 1
line_type = 2


def find_timestamp_jumps(index: pd.DatetimeIndex) -> None:

        delta = index[1:] - index[:-1]

        index = pd.Series(index)

        for i in range(0, len(index) - 1):
            if delta[i] >= pd.Timedelta(10, unit='s'):
                # print(f'Found timestamp jump at {i} between '
                #         f'timestamps {index[i]} and {index[i+1]}')
                print(index[i])
                print(index[i+1])
            

def create_channel_names(median_labels, stdev_labels):

    median_labels = ['m_{}'.format(median_label)\
                        for median_label in median_labels]

    stdev_labels = ['std_{}'.format(stdev_label)
                        for stdev_label in stdev_labels]

    labels = np.concatenate((median_labels,
                                stdev_labels))

    return labels


def fig_to_numpy_array(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    buf = np.array(fig.canvas.renderer.buffer_rgba())

    return cv.cvtColor(buf,cv.COLOR_RGBA2BGR)


if __name__ == '__main__':

    np.random.seed(42)

    parser = argparse.ArgumentParser(description='Eclipse Dataset Generator')

    parser.add_argument('--dataset-dir', type=str, default='../datasets/eclipse')
    parser.add_argument('--generate-videos', action='store_true')
    parser.add_argument('--video-output-dir', type=str, default='../videos')
    
    args = parser.parse_args()

    # Load datasets

    train_set_x_df = pd.read_hdf(f'{args.dataset_dir}/prod_train_data.hdf')
    train_set_y_df = pd.read_csv(f'{args.dataset_dir}/prod_train_label.csv')

    train_set_x_df['job_id'] =\
        pd.to_numeric(train_set_x_df['job_id'])
    train_set_x_df['component_id'] =\
        pd.to_numeric(train_set_x_df['component_id'])
    
    test_set_x_df = pd.read_hdf(f'{args.dataset_dir}/prod_test_data.hdf')
    test_set_y_df = pd.read_csv(f'{args.dataset_dir}/prod_test_label.csv')

    test_set_x_df['job_id'] =\
        pd.to_numeric(test_set_x_df['job_id'])
    test_set_x_df['component_id'] =\
        pd.to_numeric(test_set_x_df['component_id'])
    

    print(f'Train set size: {len(train_set_x_df)}')
    print(f'Test set size: {len(test_set_x_df)}')

    column_names_train = list((train_set_x_df).columns.values)
    column_names_test = list((test_set_x_df).columns.values)

    print(f'Channels train: {len(column_names_train)}')
    print(f'Channels test: {len(column_names_test)}')

    label_indices = ['job_id', 'component_id']

    train_set_x_df.set_index(label_indices, inplace=True)
    train_set_y_df.set_index(label_indices, inplace=True)

    test_set_x_df.set_index(label_indices, inplace=True)
    test_set_y_df.set_index(label_indices, inplace=True)

    train_set_x_df['app_name'] = train_set_y_df['app_name']
    train_set_x_df['binary_anom'] = train_set_y_df['binary_anom']

    test_set_x_df['app_name'] = test_set_y_df['app_name']
    train_set_x_df['binary_anom'] = train_set_y_df['binary_anom']

    train_set_x_df.reset_index(inplace=True)
    test_set_x_df.reset_index(inplace=True)

    train_set_x_df['id'] =\
        train_set_x_df[['component_id', 'job_id']]\
            .agg(lambda x: f'{x[0]}_{x[1]}', axis=1)

    test_set_x_df['id'] =\
        test_set_x_df[['component_id', 'job_id']]\
            .agg(lambda x: f'{x[0]}_{x[1]}', axis=1)

    train_set_x_df.drop(['component_id', 'job_id'],
                                axis=1, inplace=True)
    test_set_x_df.drop(['component_id', 'job_id'],
                                axis=1, inplace=True)

    data_indices = ['timestamp', 'app_name', 'id']

    train_set_x_df.set_index(data_indices, inplace=True)
    test_set_x_df.set_index(data_indices, inplace=True)

    app_names = train_set_y_df['app_name'].unique()

    print('Train\n')

    for app_name in app_names:

        per_app_data = train_set_x_df.xs(app_name, level=1)

        ids = per_app_data.reset_index()['id'].unique()

        print(f'{app_name}: {len(per_app_data)}')

        lengths = []
        starts = []
        ends = []

        length_max = 0
        id_length_max = ''

        start_min = np.iinfo(np.int64).max
        id_start_min = ''

        end_max = 0
        id_end_max = ''

        for id_ in ids:

            per_instance_data = per_app_data.xs(id_, level=1)

            print(f'ID: {id_}: {len(per_instance_data)}')

            timestamps = per_instance_data.reset_index()['timestamp']

            timestamps = pd.Index(timestamps)

            delta = timestamps[1:] - timestamps[:-1]
            delta = pd.Series(delta)

            print(f'Delta mean: {delta.mean():.5f}\tmedian: {delta.median():.5f}'\
                                        f'\tmin: {delta.min()}\tmax: {delta.max()}')

            lengths.append(len(timestamps))

            if len(timestamps) > length_max:
                length_max = len(timestamps)
                id_length_max = id_

            starts.append(timestamps[0])
            
            if timestamps[0] < start_min:
                start_min = timestamps[0]
                id_start_min = id_

            ends.append(timestamps[-1])

            if timestamps[-1] > end_max:
                end_max = timestamps[-1]
                id_end_max = id_

            # print(f'Timestamp range: {timestamps.min()} - {timestamps.max()}')

        print(f'Earliest timestamp at: {id_start_min}'
                            f'\ttimestamp: {start_min}'\
                            f'\tstart time mean: {np.mean(starts):.3f}'\
                            f'\tstart time std: {np.std(starts):.3f}')

        print(f'Latest timestamp at: {id_end_max}'\
                            f'\ttimestamp: {end_max}'\
                            f'\tend time mean: {np.mean(ends):.3f}'\
                            f'\tend time std: {np.std(ends):.3f}')

        print(f'Longest series at: {id_length_max}'\
                            f'\tlength: {length_max}'\
                            f'\tlength mean: {np.mean(lengths):.3f}'\
                            f'\tlength std: {np.std(lengths):.3f}')

    print('\nTest\n')

    for app_name in app_names:

        per_app_data = test_set_x_df.xs(app_name, level=1)

        ids = per_app_data.reset_index()['id'].unique()

        print(f'{app_name}: {len(per_app_data)}')

        lengths = []
        starts = []
        ends = []

        length_max = 0
        id_length_max = ''

        start_min = np.iinfo(np.int64).max
        id_start_min = ''

        end_max = 0
        id_end_max = ''

        for id_ in ids:

            per_instance_data = per_app_data.xs(id_, level=1)

            print(f'ID: {id_}: {len(per_instance_data)}')

            timestamps = per_instance_data.reset_index()['timestamp']

            timestamps = pd.Index(timestamps)

            delta = timestamps[1:] - timestamps[:-1]
            delta = pd.Series(delta)

            print(f'Delta mean: {delta.mean():.5f}\tmedian: {delta.median():.5f}'\
                                        f'\tmin: {delta.min()}\tmax: {delta.max()}')

            lengths.append(len(timestamps))

            if len(timestamps) > length_max:
                length_max = len(timestamps)
                id_length_max = id_
            
            starts.append(timestamps[0])

            if timestamps[0] < start_min:
                start_min = timestamps[0]
                id_start_min = id_

            ends.append(timestamps[-1])

            if timestamps[-1] > end_max:
                end_max = timestamps[-1]
                id_end_max = id_

        print(f'Earliest timestamp at: {id_start_min}'
                            f'\ttimestamp: {start_min}'\
                            f'\tstart time mean: {np.mean(starts):.3f}'\
                            f'\tstart time std: {np.std(starts):.3f}')

        print(f'Latest timestamp at: {id_end_max}'\
                            f'\ttimestamp: {end_max}'\
                            f'\tend time mean: {np.mean(ends):.3f}'\
                            f'\tend time std: {np.std(ends):.3f}')

        print(f'Longest series at: {id_length_max}'\
                            f'\tlength: {length_max}'\
                            f'\tlength mean: {np.mean(lengths):.3f}'\
                            f'\tlength std: {np.std(lengths):.3f}')

    exit()

    # Unlabeled train set

    # Reduce dataset

    rack_data_train_unlabeled_all = []

    columns_reduced_train_unlabeled = None
    keys_last = None

    train_set_unlabeled_x_df = train_set_x_df

    print(f'Train set size total: {len(train_set_x_df)}')

    for count, row_x_data in enumerate(tqdm(train_set_unlabeled_x_df.to_numpy(),
                                                desc='Generating unlabeled train set')):

        rack_buckets_data = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_train[index]].append(datapoint)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys(),\
                                                    'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_train_unlabeled) == type(None):
            columns_reduced_train_unlabeled = create_channel_names(rack_median_dcm_rates.keys(),
                                                                    rack_dcm_rate_stdevs.keys())

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

        rack_data_train_unlabeled_all.append(rack_data_np)

    rack_data_train_unlabeled_all_np = np.stack(rack_data_train_unlabeled_all)
    rack_data_train_unlabeled_all_np = np.nan_to_num(rack_data_train_unlabeled_all_np, nan=-1)

    nan_amount_train_unlabeled = 100*pd.isna(rack_data_train_unlabeled_all_np.flatten()).sum()/\
                                                            rack_data_train_unlabeled_all_np.size

    print('NaN amount reduced train set: {:.3f} %'.format(nan_amount_train_unlabeled))

    # Save dataset

    train_set_unlabeled_x_df = pd.DataFrame(rack_data_train_unlabeled_all_np,
                                                        train_set_unlabeled_x_df.index,
                                                        columns_reduced_train_unlabeled)

    train_set_unlabeled_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'\
                                            f'train_set_{args.variant}_x.h5',
                                        key='reduced_hlt_train_set_x',
                                        mode='w')

    if args.generate_videos:

        four_cc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'\
                                            f'train_set_{args.variant}.mp4',
                                        four_cc, 60, (image_width, image_height))

        for count in tqdm(range(len(rack_data_train_unlabeled_all_np)),
                            desc='Generating unlabeled train set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_train_unlabeled_all_np[lower_bound:count, :])\
                                if len(rack_data_train_unlabeled_all_np[lower_bound:count, :])\
                            else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_train_unlabeled_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Labeled train set

    test_set_size = len(test_set_x_df)

    train_set_labeled_x_df = test_set_x_df.iloc[:test_set_size//4, :]

    for count in range(1, len(train_set_labeled_x_df.index)):
        if train_set_labeled_x_df.index[count] <=\
                train_set_labeled_x_df.index[count-1]:
            print(f'Non-monotonic timestamp increase at {count-1}:\t'
                    f'First timestamp: {train_set_labeled_x_df.index[count-1]}\t'
                     f'Second timestamp: {train_set_labeled_x_df.index[count]}')

    column_names = train_set_labeled_x_df.columns
    timestamps = train_set_labeled_x_df.index

    # Generate labels for actual anomalies

    labels = generate_anomaly_labels(tpu_failure_log_df,
                                                timestamps,
                                                column_names,
                                                np.array(tpu_numbers_test),
                                                prepad=5).to_numpy()
    
    # Generate synthetic anomalies and corresponding labels

    anomaly_generator_train_labeled = MultivariateDataGenerator(train_set_labeled_x_df,
                                                                                labels,
                                                                                window_size_min=16,
                                                                                window_size_max=256)

    anomaly_generator_train_labeled.point_global_outliers(rack_count=3,
                                                                ratio=0.001,
                                                                factor=0.5,
                                                                radius=50)
    
    anomaly_generator_train_labeled.point_contextual_outliers(rack_count=3,
                                                                    ratio=0.001,
                                                                    factor=0.5,
                                                                    radius=50)

    anomaly_generator_train_labeled.persistent_global_outliers(rack_count=3,
                                                                    ratio=0.01,
                                                                    factor=1,
                                                                    radius=50)
    
    anomaly_generator_train_labeled.persistent_contextual_outliers(rack_count=3,
                                                                        ratio=0.005,
                                                                        factor=0.5,
                                                                        radius=50)

    anomaly_generator_train_labeled.collective_global_outliers(rack_count=3,
                                                                    ratio=0.005,
                                                                    option='square',
                                                                    coef=5,
                                                                    noise_amp=0.5,
                                                                    level=10,
                                                                    freq=0.1)

    anomaly_generator_train_labeled.collective_trend_outliers(rack_count=3,
                                                                    ratio=0.005,
                                                                    factor=0.5)

    # Reduce dataset and labels

    dataset = anomaly_generator_train_labeled.get_dataset_np()
    
    labels = remove_undetectable_anomalies(
                        np.nan_to_num(dataset),
                        anomaly_generator_train_labeled.get_labels_np())

    rack_data_train_labeled_all = []
    rack_labels_train_labeled_all = []

    columns_reduced_train_labeled = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(dataset, labels),
                                total=len(dataset),
                                desc='Generating labeled train set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_test[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_test[index]].append(label)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert (rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys()) and\
                                (rack_median_dcm_rates.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_train_labeled) == type(None):
            columns_reduced_train_labeled = create_channel_names(rack_median_dcm_rates.keys(),
                                                            rack_dcm_rate_stdevs.keys())
            
            assert np.array_equal(columns_reduced_train_labeled, columns_reduced_train_unlabeled),\
                                            "Labeled train columns don't match unlabeled train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

        rack_data_train_labeled_all.append(rack_data_np)

        rack_labels_train_labeled_all.append(np.array(list(rack_labels.values())))

    rack_data_train_labeled_all_np = np.stack(rack_data_train_labeled_all)
    rack_data_train_labeled_all_np = np.nan_to_num(rack_data_train_labeled_all_np, nan=-1)

    nan_amount_train_labeled = 100*pd.isna(rack_data_train_labeled_all_np.flatten()).sum()/\
                                                            rack_data_train_labeled_all_np.size

    print('NaN amount reduced labeled train set: {:.3f} %'.format(nan_amount_train_labeled))

    rack_labels_train_labeled_all_np = np.stack(rack_labels_train_labeled_all)

    rack_labels_train_labeled_all_np = np.concatenate([rack_labels_train_labeled_all_np,\
                                                        rack_labels_train_labeled_all_np],
                                                        axis=1)
    
    # Save dataset and labels

    train_set_labeled_x_df = pd.DataFrame(rack_data_train_labeled_all_np,
                                                                timestamps,
                                                                columns_reduced_train_labeled)

    train_set_labeled_y_df = pd.DataFrame(rack_labels_train_labeled_all_np,
                                                                timestamps,
                                                                columns_reduced_train_labeled)

    anomalies_per_column = np.count_nonzero(rack_labels_train_labeled_all_np, axis=0)

    anomaly_ratio_per_column = anomalies_per_column/\
                                len(rack_labels_train_labeled_all_np)

    for anomalies, anomaly_ratio, column_name in zip(anomalies_per_column,
                                                        anomaly_ratio_per_column,
                                                        columns_reduced_train_labeled):

        print(f'{column_name}: {anomalies} anomalies, {100*anomaly_ratio} % of all data')

    train_set_labeled_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'\
                                        f'labeled_train_set_{args.variant}_x.h5',
                                    key='reduced_hlt_labeled_train_set_x',
                                    mode='w')

    train_set_labeled_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'\
                                        f'labeled_train_set_{args.variant}_y.h5',
                                    key='reduced_hlt_labeled_train_set_y',
                                    mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'
                                        f'labeled_train_set_{args.variant}.mp4',
                                    four_cc, 60, (image_width, image_height))


        for count in tqdm(range(len(rack_data_train_labeled_all_np)),
                        desc='Generating labeled train set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_train_labeled_all_np[lower_bound:count, :])\
                                if len(rack_data_train_labeled_all_np[lower_bound:count, :])\
                                else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_train_labeled_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Unreduced test set

    column_names = test_set_x_df.columns
    timestamps = test_set_x_df.index

    # Generate labels for actual anomalies

    labels_actual = generate_anomaly_labels(tpu_failure_log_df,
                                                    timestamps,
                                                    column_names,
                                                    np.array(tpu_numbers_test),
                                                    prepad=5).to_numpy()

    # Generate synthetic anomalies and corresponding labels

    anomaly_generator_test = MultivariateDataGenerator(test_set_x_df,
                                                            labels_actual,
                                                            window_size_min=16,
                                                            window_size_max=256)

    anomaly_generator_test.point_global_outliers(rack_count=3,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=50)
    
    anomaly_generator_test.point_contextual_outliers(rack_count=3,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=50)

    anomaly_generator_test.persistent_global_outliers(rack_count=3,
                                                            ratio=0.005,
                                                            factor=0.5,
                                                            radius=50)
    
    anomaly_generator_test.persistent_contextual_outliers(rack_count=3,
                                                                ratio=0.005,
                                                                factor=0.5,
                                                                radius=50)

    anomaly_generator_test.collective_global_outliers(rack_count=3,
                                                            ratio=0.005,
                                                            option='square',
                                                            coef=5,
                                                            noise_amp=0.05,
                                                            level=10,
                                                            freq=0.1)

    anomaly_generator_test.collective_trend_outliers(rack_count=3,
                                                            ratio=0.005,
                                                            factor=0.5)

    anomaly_generator_test.intra_rack_outliers(ratio_temporal=0.001,
                                                    ratio_channels=0.05,
                                                    average_duration=10.,
                                                    stdev_duration=1.)

    labels_unreduced =\
        remove_undetectable_anomalies(
            np.nan_to_num(anomaly_generator_test.get_dataset_np()),
            anomaly_generator_test.get_labels_np())

    # Save dataset and labels

    test_set_unreduced_x_df =\
        pd.DataFrame(anomaly_generator_test.get_dataset_np(),
                        anomaly_generator_test.get_timestamps_pd(),
                        test_set_x_df.columns)

    test_set_unreduced_y_df =\
        pd.DataFrame(labels_unreduced,
                        anomaly_generator_test.get_timestamps_pd(),
                        test_set_x_df.columns)

    test_set_unreduced_x_df.to_hdf(
            f'{args.dataset_dir}/unreduced_hlt_test_set_{args.variant}_x.h5',
            key='unreduced_hlt_test_set_x', mode='w')

    test_set_unreduced_y_df.to_hdf(
            f'{args.dataset_dir}/unreduced_hlt_test_set_{args.variant}_y.h5',
            key='unreduced_hlt_test_set_y', mode='w')
    
    # Reduced test set

    # Generate synthetic anomalies and corresponding labels

    anomaly_generator_test = MultivariateDataGenerator(test_set_x_df,
                                                            labels_actual,
                                                            window_size_min=16,
                                                            window_size_max=256)

    anomaly_generator_test.point_global_outliers(rack_count=3,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=50)
    
    anomaly_generator_test.point_contextual_outliers(rack_count=3,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=50)

    anomaly_generator_test.persistent_global_outliers(rack_count=3,
                                                            ratio=0.005,
                                                            factor=0.5,
                                                            radius=50)
    
    anomaly_generator_test.persistent_contextual_outliers(rack_count=3,
                                                                ratio=0.005,
                                                                factor=0.5,
                                                                radius=50)

    anomaly_generator_test.collective_global_outliers(rack_count=3,
                                                            ratio=0.005,
                                                            option='square',
                                                            coef=5,
                                                            noise_amp=0.05,
                                                            level=10,
                                                            freq=0.1)

    anomaly_generator_test.collective_trend_outliers(rack_count=3,
                                                            ratio=0.005,
                                                            factor=0.5)

    # Reduce dataset and labels

    dataset = anomaly_generator_test.get_dataset_np()

    labels = remove_undetectable_anomalies(
                        np.nan_to_num(dataset),
                        anomaly_generator_test.get_labels_np())

    rack_data_test_all = []
    rack_labels_test_all = []

    columns_reduced_test = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(dataset, labels),
                                total=len(dataset),
                                desc='Generating test set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_test[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_test[index]].append(label)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                            'Rack bucket keys changed between slices'

            assert (rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys()) and\
                                (rack_median_dcm_rates.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_test) == type(None):
            columns_reduced_test = create_channel_names(rack_median_dcm_rates.keys(),
                                                            rack_dcm_rate_stdevs.keys())
            
            assert np.array_equal(columns_reduced_test, columns_reduced_train_unlabeled),\
                                                    "Test columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

        rack_data_test_all.append(rack_data_np)

        rack_labels_test_all.append(np.array(list(rack_labels.values())))

    rack_data_test_all_np = np.stack(rack_data_test_all)
    rack_data_test_all_np = np.nan_to_num(rack_data_test_all_np, nan=-1)

    nan_amount_test = 100*pd.isna(rack_data_test_all_np.flatten()).sum()/\
                                                    rack_data_test_all_np.size

    print('NaN amount reduced test set: {:.3f} %'.format(nan_amount_test))

    rack_labels_test_all_np = np.stack(rack_labels_test_all)

    rack_labels_test_all_np = np.concatenate([rack_labels_test_all_np,\
                                                rack_labels_test_all_np],
                                                axis=1)
    
    # Save dataset and labels

    test_set_reduced_x_df = pd.DataFrame(rack_data_test_all_np,
                                            anomaly_generator_test.get_timestamps_pd(),
                                            columns_reduced_test)

    test_set_reduced_y_df = pd.DataFrame(rack_labels_test_all_np,
                                            anomaly_generator_test.get_timestamps_pd(),
                                            columns_reduced_test)

    anomalies_per_column = np.count_nonzero(rack_labels_test_all_np, axis=0)

    anomaly_ratio_per_column = anomalies_per_column/\
                                    len(rack_labels_test_all_np)

    for anomalies, anomaly_ratio, column_name in zip(anomalies_per_column,
                                                        anomaly_ratio_per_column,
                                                        columns_reduced_test):

        print(f'{column_name}: {anomalies} anomalies, {100*anomaly_ratio} % of all data')

    test_set_reduced_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                        f'test_set_{args.variant}_x.h5',
                                    key='reduced_hlt_test_set_x',
                                    mode='w')

    test_set_reduced_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                        f'test_set_{args.variant}_y.h5',
                                    key='reduced_hlt_test_set_y',
                                    mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'
                                        f'test_set_{args.variant}.mp4',
                                    four_cc, 60, (image_width, image_height))

        for count in tqdm(range(len(rack_data_test_all_np)),
                                    desc='Generating test set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_test_all_np[lower_bound:count, :])\
                                if len(rack_data_test_all_np[lower_bound:count, :])\
                                else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_test_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Clean val set

    # Reduce dataset

    clean_val_set_x_df = val_set_x_df

    column_names = clean_val_set_x_df.columns
    timestamps = clean_val_set_x_df.index

    rack_data_clean_val_all = []

    columns_reduced_clean_val = None
    keys_last = None

    for count, row_x_data in enumerate(tqdm(val_set_x_df.to_numpy(),
                                                desc='Generating clean val set')):

        rack_buckets_data = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_val[index]].append(datapoint)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys(),\
                                                    'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_clean_val) == type(None):
            columns_reduced_clean_val = create_channel_names(rack_median_dcm_rates.keys(),
                                                                rack_dcm_rate_stdevs.keys())

            assert np.array_equal(columns_reduced_clean_val, columns_reduced_train_unlabeled),\
                                                        "Val columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

        rack_data_clean_val_all.append(rack_data_np)

    rack_data_clean_val_all_np = np.stack(rack_data_clean_val_all)
    rack_data_clean_val_all_np = np.nan_to_num(rack_data_clean_val_all_np, nan=-1)

    nan_amount_clean_val = 100*pd.isna(rack_data_clean_val_all_np.flatten()).sum()/\
                                                    rack_data_clean_val_all_np.size

    print('NaN amount reduced clean val set: {:.3f} %'.format(nan_amount_clean_val))

    # Save dataset

    clean_val_set_x_df = pd.DataFrame(rack_data_clean_val_all_np,
                                                val_set_x_df.index,
                                                columns_reduced_clean_val)

    clean_val_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                    f'clean_val_set_{args.variant}_x.h5',
                                key='reduced_hlt_clean_val_set_x',
                                mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'
                                        f'clean_val_set_{args.variant}.mp4',
                                    four_cc, 60, (image_width, image_height))

        for count in tqdm(range(len(rack_data_clean_val_all_np)),
                                    desc='Generating clean val set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_clean_val_all_np[lower_bound:count, :])\
                                    if len(rack_data_clean_val_all_np[lower_bound:count, :])\
                                    else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_clean_val_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()

    # Dirty val set

    val_set_x_df = pd.concat((val_set_x_df.iloc[:9270, :],
                                test_set_x_df.iloc[-8570:, :]))

    column_names_val = list((val_set_x_df).columns.values)
    tpu_numbers_val = [get_tpu_number(label) for label in column_names_val]
    tpu_numbers_val_unique = np.array(list(set(tpu_numbers_val)))
    rack_numbers_val = np.floor_divide(tpu_numbers_val, 1000)

    for count in range(1, len(val_set_x_df.index)):
        if val_set_x_df.index[count] <=\
                val_set_x_df.index[count-1]:
            print(f'Non-monotonic timestamp increase at {count-1}:\t'
                    f'First timestamp: {val_set_x_df.index[count-1]}\t'
                     f'Second timestamp: {val_set_x_df.index[count]}')

    column_names = val_set_x_df.columns
    timestamps = val_set_x_df.index

    # Generate labels for actual anomalies

    labels_actual = generate_anomaly_labels(tpu_failure_log_df,
                                                    timestamps,
                                                    column_names,
                                                    np.array(tpu_numbers_val),
                                                    prepad=5).to_numpy()
    
    # Generate synthetic anomalies and corresponding labels

    anomaly_generator_val = MultivariateDataGenerator(val_set_x_df,
                                                        labels_actual,
                                                        window_size_min=16,
                                                        window_size_max=256)

    anomaly_generator_val.point_global_outliers(rack_count=3,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=50)
    
    anomaly_generator_val.point_contextual_outliers(rack_count=3,
                                                        ratio=0.001,
                                                        factor=0.5,
                                                        radius=50)

    anomaly_generator_val.persistent_global_outliers(rack_count=3,
                                                            ratio=0.005,
                                                            factor=0.5,
                                                            radius=50)
    
    anomaly_generator_val.persistent_contextual_outliers(rack_count=3,
                                                                ratio=0.005,
                                                                factor=0.5,
                                                                radius=50)

    anomaly_generator_val.collective_global_outliers(rack_count=3,
                                                        ratio=0.005,
                                                        option='square',
                                                        coef=5,
                                                        noise_amp=0.5,
                                                        level=10,
                                                        freq=0.1)

    anomaly_generator_val.collective_trend_outliers(rack_count=3,
                                                        ratio=0.005,
                                                        factor=0.5)
    
    # Reduce dataset and labels
    
    dataset = anomaly_generator_val.get_dataset_np()

    labels = remove_undetectable_anomalies(
                        np.nan_to_num(dataset),
                        anomaly_generator_val.get_labels_np())

    rack_data_val_all = []
    rack_labels_val_all = []

    columns_reduced_val = None
    keys_last = None

    for count, (row_x_data, row_x_labels)\
            in enumerate(tqdm(zip(dataset, labels),
                                    total=len(dataset),
                                    desc='Generating dirty val set')):

        rack_buckets_data = defaultdict(list)
        rack_buckets_labels = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            rack_buckets_data[rack_numbers_val[index]].append(datapoint)

        for index, label in enumerate(row_x_labels):
            rack_buckets_labels[rack_numbers_val[index]].append(label)

        rack_median_dcm_rates = {}
        rack_dcm_rate_stdevs = {}
        rack_labels = {}

        for rack, rack_bucket in rack_buckets_data.items():
            rack_median_dcm_rates[rack] = np.nanmedian(rack_bucket)
            rack_dcm_rate_stdevs[rack] = np.nanstd(rack_bucket)

        for rack, rack_bucket in rack_buckets_labels.items():

            rack_label = 0

            for label in rack_bucket:
                rack_label = rack_label | label
                
            rack_labels[rack] = rack_label

        rack_median_dcm_rates = dict(sorted(rack_median_dcm_rates.items()))
        rack_dcm_rate_stdevs = dict(sorted(rack_dcm_rate_stdevs.items()))

        rack_labels = dict(sorted(rack_labels.items()))

        if keys_last != None:
            assert rack_median_dcm_rates.keys() == keys_last,\
                                                    'Rack bucket keys changed between slices'

            assert (rack_median_dcm_rates.keys() == rack_dcm_rate_stdevs.keys()) and\
                                (rack_median_dcm_rates.keys() == rack_labels.keys()),\
                                                        'Rack bucket keys not identical'

        keys_last = rack_median_dcm_rates.keys()

        if type(columns_reduced_val) == type(None):
            columns_reduced_val = create_channel_names(rack_median_dcm_rates.keys(),
                                                        rack_dcm_rate_stdevs.keys())

            assert np.array_equal(columns_reduced_val, columns_reduced_train_unlabeled),\
                                                    "Val columns don't match train columns" 

        rack_data_np = np.concatenate((np.array(list(rack_median_dcm_rates.values())),
                                            np.array(list(rack_dcm_rate_stdevs.values()))))

        rack_data_val_all.append(rack_data_np)

        rack_labels_val_all.append(np.array(list(rack_labels.values())))

    rack_data_val_all_np = np.stack(rack_data_val_all)
    rack_data_val_all_np = np.nan_to_num(rack_data_val_all_np, nan=-1)

    nan_amount_dirty_val = 100*pd.isna(rack_data_val_all_np.flatten()).sum()/\
                                                    rack_data_val_all_np.size

    print('NaN amount reduced dirty val set: {:.3f} %'.format(nan_amount_dirty_val))

    rack_labels_val_all_np = np.stack(rack_labels_val_all)

    rack_labels_val_all_np = np.concatenate([rack_labels_val_all_np,\
                                                rack_labels_val_all_np],
                                                axis=1)

    val_set_x_df = pd.DataFrame(rack_data_val_all_np,
                                    anomaly_generator_val.get_timestamps_pd(),
                                    columns_reduced_val)

    val_set_y_df = pd.DataFrame(rack_labels_val_all_np,
                                    anomaly_generator_val.get_timestamps_pd(),
                                    columns_reduced_val)

    anomalies_per_column = np.count_nonzero(rack_labels_val_all_np, axis=0)

    anomaly_ratio_per_column = anomalies_per_column/\
                                    len(rack_labels_val_all_np)
    
    # Save dataset and labels

    val_set_x_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                f'val_set_{args.variant}_x.h5',
                            key='reduced_hlt_val_set_x',
                            mode='w')

    val_set_y_df.to_hdf(f'{args.dataset_dir}/reduced_hlt_'
                                f'val_set_{args.variant}_y.h5',
                            key='reduced_hlt_val_set_y',
                            mode='w')

    if args.generate_videos:

        writer = cv.VideoWriter(f'{args.video_output_dir}/reduced_hlt_'
                                            f'val_set_{args.variant}.mp4',
                                    four_cc, 60,(image_width, image_height))

        for count in tqdm(range(len(rack_data_val_all_np)),
                            desc='Generating dirty val set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(rack_data_val_all_np[lower_bound:count, :])\
                                if len(rack_data_val_all_np[lower_bound:count, :])\
                                else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-Rack Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                rack_data_val_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()


