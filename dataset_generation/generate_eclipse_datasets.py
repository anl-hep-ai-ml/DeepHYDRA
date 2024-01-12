#!/usr/bin/env python3
import math
import re
import argparse
from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
import prince
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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


def get_contiguous_runs(x):
    '''
    Find runs of consecutive items in an array.
    As published in https://gist.github.com/alimanfoo/c5977e87111abe8127453b21204c1065
    '''

    # Ensure array

    x = np.asanyarray(x)

    if x.ndim != 1:
        raise ValueError('Only 1D arrays supported')

    n = x.shape[0]

    # Handle empty array

    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:

        # Find run starts

        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True

        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # Find run values
        run_values = x[loc_run_start]

        # Find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        run_starts = np.compress(run_values, run_starts)
        run_lengths = np.compress(run_values, run_lengths)

        run_ends = run_starts + run_lengths

        return run_starts, run_ends


def remove_timestamp_jumps(index: pd.Index) -> pd.Index:

        delta = index[1:] - index[:-1]

        index = pd.Series(index)

        for i in range(1, len(index)):
            if delta[i - 1] > 1:
                index[i:] = index[i:] - delta[i - 1] + 1

        return pd.Index(index)


def run_pca(data_df: pd.DataFrame,
                cols_without_contribution_all):

    data_np = data_df.to_numpy(dtype=np.float64)
    scaler = StandardScaler()

    scaler.fit(data_np)

    data_scaled_np =\
        scaler.transform(data_np)

    dataset_df = pd.DataFrame(data_scaled_np,
                                index=data_df.index,
                                columns=data_df.columns)

    pca = prince.PCA(n_components=32, 
                            n_iter=10,
                            copy=True,
                            check_input=True,
                            random_state=42,
                            engine='sklearn')

    pca = pca.fit(data_df)

    # print('Eigenvalue summary:')
    # print(pca.eigenvalues_summary)

    cols_without_contribution = []

    for index, data in pca.column_contributions_.iterrows():
        if data.sum() <= 0.01:
            cols_without_contribution.append(index)

    if cols_without_contribution_all is None:
        cols_without_contribution_all = cols_without_contribution
    else:
        cols_without_contribution_all =\
            np.intersect1d(cols_without_contribution_all,
                                cols_without_contribution)

    return cols_without_contribution_all


def rename_columns(columns: pd.Index, 
                            app_name: str,
                            id: str):
    
    columns = pd.Series(columns)

    def _renaming_func(element):
        if element != 'label':
            constituents = str(element).split('::')
            name = f'{constituents[1].lower()}_{constituents[0].lower()}'
        else:
            name = element

        return name + f'_{app_name}_{id}'

    return pd.Index(columns.map(_renaming_func))


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

    timestamps_train = train_set_x_df.reset_index()['timestamp']
    timestamps_test = test_set_x_df.reset_index()['timestamp']
    
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

    # Run PCA and drop columns that don't contribute
    # to dataset variation

    cols_without_contribution_all = None

    # cols_without_contribution_all =\
    #                 run_pca(train_set_x_df,
    #                             cols_without_contribution_all)

    # cols_without_contribution_all =\
    #                 run_pca(test_set_x_df,
    #                             cols_without_contribution_all)

    dataset_combined_df = pd.concat((train_set_x_df,
                                        test_set_x_df),
                                        ignore_index=True)

    cols_without_contribution_all =\
                    run_pca(dataset_combined_df,
                                cols_without_contribution_all)

    amount_relative = len(cols_without_contribution_all)/\
                                    len(train_set_x_df.columns)

    print('Cols without contribution')
    print(f'Amount: {len(cols_without_contribution_all)}')
    print(f'Relative amount: {100*amount_relative:.3f} %')
    
    train_set_x_df.drop(cols_without_contribution_all,
                                    axis=1, inplace=True)
    test_set_x_df.drop(cols_without_contribution_all,
                                    axis=1, inplace=True)

    train_set_x_df['app_name'] = train_set_y_df['app_name']
    test_set_x_df['app_name'] = test_set_y_df['app_name']
    
    train_set_x_df['label'] = 0

    train_set_x_df['label'] =\
        train_set_x_df['label'].astype(np.uint8)

    label_map = {'none': 0b00,
                    'cpuoccupy': 0b01,
                    'memleak': 0b10}

    test_set_x_df['label'] =\
        test_set_y_df['anom_name'].map(label_map).astype(np.uint8)

    train_set_x_df.reset_index(inplace=True)
    test_set_x_df.reset_index(inplace=True)

    # train_set_x_df['id'] =\
    #     train_set_x_df[['component_id', 'job_id']]\
    #         .agg(lambda x: f'{x[0]}_{x[1]}', axis=1)

    # test_set_x_df['id'] =\
    #     test_set_x_df[['component_id', 'job_id']]\
    #         .agg(lambda x: f'{x[0]}_{x[1]}', axis=1)

    train_set_x_df['id'] = train_set_x_df['component_id']
    test_set_x_df['id'] = test_set_x_df['component_id']

    train_set_x_df.drop(['component_id', 'job_id'],
                                axis=1, inplace=True)
    test_set_x_df.drop(['component_id', 'job_id'],
                                axis=1, inplace=True)

    data_indices = ['timestamp', 'app_name', 'id']
    # data_indices = ['timestamp', 'app_name', 'component_id', 'job_id']

    train_set_x_df.set_index(data_indices, inplace=True)
    test_set_x_df.set_index(data_indices, inplace=True)
    
    # Reshape datasets

    app_names = train_set_y_df['app_name'].unique()

    train_subsets = defaultdict(list)

    # print('Train\n')

    for app_name in app_names:

        per_app_data = train_set_x_df.xs(app_name, level=1)

        ids = per_app_data.reset_index()['id'].unique()

        # print(f'{app_name}: {len(per_app_data)}')

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

            # print(f'ID: {id_}: {len(per_instance_data)}')

            timestamps = per_instance_data.reset_index()['timestamp']

            timestamps = pd.Index(timestamps)

            delta = timestamps[1:] - timestamps[:-1]
            delta = pd.Series(delta)

            # print(f'Delta mean: {delta.mean():.5f}\tmedian: {delta.median():.5f}'\
            #                             f'\tmin: {delta.min()}\tmax: {delta.max()}')

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

            per_instance_data.columns =\
                    rename_columns(per_instance_data.columns,
                                                app_name, id_)

            train_subsets[id_].append(per_instance_data)

    # train_set_reshaped_df = pd.concat(train_subsets, axis=1)

    # print(train_set_reshaped_df)

    # nan_amount = np.mean(np.sum(pd.isna(train_set_reshaped_df.to_numpy()), 1)/\
    #                                                     train_set_reshaped_df.shape[1])

    # print(f'Mean sparsity reshaped train set: {100*nan_amount:.3f} %')

    app_names = test_set_y_df['app_name'].unique()

    test_subsets_in = defaultdict(list)

    # print('\nTest\n')

    for app_name in app_names:

        per_app_data = test_set_x_df.xs(app_name, level=1)

        ids = per_app_data.reset_index()['id'].unique()

        # print(f'{app_name}: {len(per_app_data)}')

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

            # print(f'ID: {id_}: {len(per_instance_data)}')

            timestamps = per_instance_data.reset_index()['timestamp']

            timestamps = pd.Index(timestamps)

            delta = timestamps[1:] - timestamps[:-1]
            delta = pd.Series(delta)

            # print(f'Delta mean: {delta.mean():.5f}\tmedian: {delta.median():.5f}'\
            #                             f'\tmin: {delta.min()}\tmax: {delta.max()}')

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

            per_instance_data.columns =\
                    rename_columns(per_instance_data.columns,
                                                app_name, id_)

            test_subsets_in[id_].append(per_instance_data)

    # test_set_reshaped_df = pd.concat(test_subsets_in, axis=1)

    # print(test_set_reshaped_df)

    # nan_amount = np.mean(np.sum(pd.isna(test_set_reshaped_df.to_numpy()), 1)/\
    #                                                     test_set_reshaped_df.shape[1])

    # print(f'Mean sparsity reshaped train set: {100*nan_amount:.3f} %')

    # Compose unlabeled train, labeled train, test,
    # unlabeled val, and labeled val datasets from
    # existing train and test sets

    output_subsets = {'unlabeled train': defaultdict(list),
                        'labeled train': defaultdict(list),
                        'test': defaultdict(list),
                        'unlabeled val': defaultdict(list),
                        'labeled val': defaultdict(list),}

    choices = ['train', 'test', 'val']

    p_train = [0.8, 0.1, 0.1]
    p_test = [0.02, 0.96, 0.02]

    rng = np.random.default_rng(42)

    for id_, data in train_subsets.items():
        for element in data:
            # data_train, data_test = train_test_split(element, test_size=0.3)
            # data_test, data_val = train_test_split(data_test, test_size=0.667)
            # data_train_labeled, _ = train_test_split(data_train, test_size=0.2)
            # data_val_labeled, _ = train_test_split(data_val, test_size=0.2)

            # unlabeled_train_subsets[id_].append(data_train)
            # labeled_train_subsets[id_].append(data_train_labeled)
            # test_subsets_out[id_].append(data_test)
            # unlabeled_val_subsets[id_].append(data_val)
            # labeled_val_subsets[id_].append(data_val_labeled)

            output_category = rng.choice(choices, p=p_train)

            if output_category == 'train':
                output_subsets['unlabeled train'][id_].append(element)

                if rng.choice(2, p=[0.1, 0.9]):
                    output_subsets['labeled train'][id_].append(element)

            elif output_category == 'test':
                output_subsets['test'][id_].append(element)

            else:
                output_subsets['unlabeled val'][id_].append(element)

                if rng.choice(2, p=[0.1, 0.9]):
                    output_subsets['labeled val'][id_].append(element)


    for id_, data in test_subsets_in.items():
        for element in data:
            # data_test, data_train_labeled = train_test_split(element, test_size=0.1)
            # data_test, data_val_labeled = train_test_split(data_test, test_size=0.1)

            # labeled_train_subsets[id_].append(data_train_labeled)
            # test_subsets_out[id_].append(data_test)
            # labeled_val_subsets[id_].append(data_val_labeled)

            output_category = rng.choice(choices, p=p_test)

            if output_category == 'train':
                output_subsets['labeled train'][id_].append(element)

            elif output_category == 'test':
                output_subsets['test'][id_].append(element)

            else:
                output_subsets['labeled val'][id_].append(element)

    # for dataset_type, dataset in output_subsets.items():

    #     print(f'{dataset_type}:')

    #     timestamp_counter = defaultdict(int)

    #     for id_, data in dataset.items():
    #         for element in data:
    #             for ts in element.index:
    #                 timestamp_counter[ts] += 1

    #     overlaps = np.array(list(timestamp_counter.values())) - 1
    #     timestamps = list(timestamp_counter.keys())

    #     overlap_starts, overlap_ends  =\
    #             get_contiguous_runs(overlaps)
        
    #     for start, end in zip(overlap_starts,
    #                                 overlap_ends):
            
    #         end = min(end, (len(overlaps) - 1))

    #         print('Found overlapping region between '\
    #                 f'{timestamps[start]} and {timestamps[end]}')

    output_datasets = {}

    for dataset_type, dataset in output_subsets.items():

        print(f'{dataset_type}:')

        subset_list = []

        for id_, subsets in dataset.items():
            for subset in subsets:
                subset_list.append(subset)

        dataset_reshaped = pd.concat(subset_list, axis=1)

        dataset_reshaped.sort_index(inplace=True)

        print(dataset_reshaped.index)

        dataset_reshaped.index =\
            remove_timestamp_jumps(dataset_reshaped.index)

        print(dataset_reshaped.index)

        # print(dataset_reshaped)

        nan_amount = np.mean(np.sum(pd.isna(dataset_reshaped.to_numpy()), 1)/\
                                                        dataset_reshaped.shape[1])

        print(f'Mean sparsity reshaped {dataset_type} set: {100*nan_amount:.3f} %')

        label_columns =\
            [col for col in dataset_reshaped.columns if 'label' in col]

        dataset_reshaped[label_columns] =\
            dataset_reshaped[label_columns].fillna(0).astype(np.uint8)

        dataset_reshaped['label'] =\
            dataset_reshaped[label_columns]\
                .agg(lambda row: reduce(lambda x, y: x|y, row.tolist()), axis=1)

        anomaly_count =\
            np.count_nonzero(dataset_reshaped['label'].to_numpy().flatten()>=1)

        print(f'Anomalous data ratio: {100*anomaly_count/len(dataset_reshaped):.3f} %')

        anomaly_ratio_cumulative =\
            np.cumsum(dataset_reshaped['label']\
                .to_numpy().flatten()>=1)/len(dataset_reshaped)
        
        if dataset_type == 'test' or dataset_type.startswith('labeled'):
            fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

            ax.set_title(f'Eclipse {dataset_type.title()} '\
                                'Set Cumulative Anomaly Ratio')
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Cumulative Anomaly Ratio')

            ax.grid()

            ax.plot(dataset_reshaped.index,
                        anomaly_ratio_cumulative)
            
            plt.tight_layout()
            plt.savefig(f"plots/eclipse_{dataset_type.replace(' ', '_')}"\
                                                    '_set_anomaly_cumsum.png')

    exit()

    # TODO: Check timestamp overlap in generated datasets

    # Unlabeled train set

    # Reduce dataset

    app_data_train_unlabeled_all = []

    columns_reduced_train_unlabeled = None
    keys_last = None

    train_set_unlabeled_x_df = train_set_x_df

    for count, row_x_data in enumerate(tqdm(train_set_unlabeled_x_df.to_numpy(),
                                                desc='Generating unlabeled train set')):

        app_buckets_data = defaultdict(list)

        for index, datapoint in enumerate(row_x_data):
            app_buckets_data[app_numbers_train[index]].append(datapoint)

        app_data_mean
        app_median_dcm_rates = {}
        app_dcm_rate_stdevs = {}

        for app, app_bucket in app_buckets_data.items():
            app_median_dcm_rates[app] = np.nanmedian(app_bucket)
            app_dcm_rate_stdevs[app] = np.nanstd(app_bucket)

        app_median_dcm_rates = dict(sorted(app_median_dcm_rates.items()))
        app_dcm_rate_stdevs = dict(sorted(app_dcm_rate_stdevs.items()))

        if keys_last != None:
            assert app_median_dcm_rates.keys() == keys_last,\
                                                    'App bucket keys changed between slices'

            assert app_median_dcm_rates.keys() == app_dcm_rate_stdevs.keys(),\
                                                    'App bucket keys not identical'

        keys_last = app_median_dcm_rates.keys()

        if type(columns_reduced_train_unlabeled) == type(None):
            columns_reduced_train_unlabeled = create_channel_names(app_median_dcm_rates.keys(),
                                                                    app_dcm_rate_stdevs.keys())

        app_data_np = np.concatenate((np.array(list(app_median_dcm_rates.values())),
                                            np.array(list(app_dcm_rate_stdevs.values()))))

        app_data_train_unlabeled_all.append(app_data_np)

    app_data_train_unlabeled_all_np = np.stack(app_data_train_unlabeled_all)
    app_data_train_unlabeled_all_np = np.nan_to_num(app_data_train_unlabeled_all_np, nan=-1)

    nan_amount_train_unlabeled = 100*pd.isna(app_data_train_unlabeled_all_np.flatten()).sum()/\
                                                            app_data_train_unlabeled_all_np.size

    print('NaN amount reduced train set: {:.3f} %'.format(nan_amount_train_unlabeled))

    # Save dataset

    train_set_unlabeled_x_df = pd.DataFrame(app_data_train_unlabeled_all_np,
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

        for count in tqdm(range(len(app_data_train_unlabeled_all_np)),
                            desc='Generating unlabeled train set animation'):

            lower_bound = max(count - plot_window_size, 0)
            upper_bound_axis = max(count, plot_window_size) + 10

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=240)

            max_val_slice = np.max(app_data_train_unlabeled_all_np[lower_bound:count, :])\
                                if len(app_data_train_unlabeled_all_np[lower_bound:count, :])\
                            else 10

            max_val_slice = min(max_val_slice, 200)

            ax.set_xlim(lower_bound, upper_bound_axis)
            ax.set_ylim(-2, max_val_slice + 10)

            ax.grid(True)

            ax.set_title("Per-app Median DCM Rates")
            ax.set_xlabel("Timestep")
            ax.set_ylabel("DCM Rate")

            ax.plot(np.arange(lower_bound, count),
                                app_data_train_unlabeled_all_np[lower_bound:count, :])

            # plt.tight_layout()

            frame = fig_to_numpy_array(fig)

            writer.write(frame)

            plt.close()

        writer.release()