#!/usr/bin/env python3
import math
import re
import argparse
from collections import defaultdict
from functools import reduce

import numpy as np
import pandas as pd
import prince
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import cv2 as cv
import matplotlib.pyplot as plt
from tqdm import tqdm

max_val = 100

image_width = 1920
image_height = 2160

plot_window_size = 100

font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (255,255,255)
thickness = 1
line_type = 2


# We convert these values to ordinal values,
# as they have less then 16 unique values over
# all the datasets

ordinal_cols = ('exa_meminfo_commitlimit_6',
                        'exa_meminfo_dirty_6',
                        'exa_meminfo_hardwarecorrupted_6',
                        'exa_meminfo_writeback_6',
                        'exa_vmstat_drop_pagecache_6',
                        'exa_vmstat_drop_slab_6',
                        'exa_vmstat_nr_anon_transparent_hugepages_6',
                        'exa_vmstat_nr_dirty_6',
                        'exa_vmstat_nr_isolated_file_6',
                        'exa_vmstat_nr_vmscan_immediate_reclaim_6',
                        'exa_vmstat_nr_vmscan_write_6',
                        'exa_vmstat_nr_writeback_6',
                        'exa_vmstat_numa_foreign_6',
                        'exa_vmstat_numa_miss_6',
                        'exa_vmstat_slabs_scanned_6',
                        'lammps_meminfo_commitlimit_6',
                        'lammps_meminfo_dirty_6',
                        'lammps_meminfo_hardwarecorrupted_6',
                        'lammps_meminfo_writeback_6',
                        'lammps_vmstat_drop_pagecache_6',
                        'lammps_vmstat_drop_slab_6',
                        'lammps_vmstat_nr_anon_transparent_hugepages_6',
                        'lammps_vmstat_nr_dirty_6',
                        'lammps_vmstat_nr_isolated_anon_6',
                        'lammps_vmstat_nr_isolated_file_6',
                        'lammps_vmstat_nr_vmscan_immediate_reclaim_6',
                        'lammps_vmstat_nr_vmscan_write_6',
                        'lammps_vmstat_nr_writeback_6',
                        'sw4_meminfo_commitlimit_6',
                        'sw4_meminfo_dirty_6',
                        'sw4_meminfo_hardwarecorrupted_6',
                        'sw4_meminfo_writeback_6',
                        'sw4_vmstat_drop_pagecache_6',
                        'sw4_vmstat_drop_slab_6',
                        'sw4_vmstat_nr_anon_transparent_hugepages_6',
                        'sw4_vmstat_nr_dirty_6',
                        'sw4_vmstat_nr_isolated_anon_6',
                        'sw4_vmstat_nr_isolated_file_6',
                        'sw4_vmstat_nr_vmscan_immediate_reclaim_6',
                        'sw4_vmstat_nr_vmscan_write_6',
                        'sw4_vmstat_nr_writeback_6',
                        'sw4_vmstat_numa_foreign_6',
                        'sw4_vmstat_numa_miss_6',
                        'sw4_vmstat_slabs_scanned_6',
                        'sw4lite_meminfo_commitlimit_6',
                        'sw4lite_meminfo_dirty_6',
                        'sw4lite_meminfo_hardwarecorrupted_6',
                        'sw4lite_meminfo_writeback_6',
                        'sw4lite_vmstat_drop_pagecache_6',
                        'sw4lite_vmstat_drop_slab_6',
                        'sw4lite_vmstat_nr_anon_transparent_hugepages_6',
                        'sw4lite_vmstat_nr_dirty_6',
                        'sw4lite_vmstat_nr_isolated_anon_6',
                        'sw4lite_vmstat_nr_isolated_file_6',
                        'sw4lite_vmstat_nr_vmscan_write_6',
                        'sw4lite_vmstat_nr_writeback_6',
                        'sw4lite_vmstat_numa_foreign_6',
                        'sw4lite_vmstat_numa_interleave_6',
                        'sw4lite_vmstat_numa_miss_6',
                        'sw4lite_vmstat_slabs_scanned_6')


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
                            id_: str):
    
    columns = pd.Series(columns)

    def _renaming_func(element):
        if element != 'label':
            constituents = str(element).split('::')
            name = f'{constituents[1].lower()}_{constituents[0].lower()}'
            name = name.replace('(', '_').replace(')', '_')
            name = name.removesuffix('_')
        else:
            name = element

        return f'{app_name}_{name}_{id_}'

    return pd.Index(columns.map(_renaming_func))


def rename_label_columns(columns: pd.Index):
    
    columns = pd.Series(columns)

    def _renaming_func(element):
        constituents = str(element).split('_')
        return f'{constituents[1]}_{constituents[0]}_{constituents[2]}'

    return pd.Index(columns.map(_renaming_func))


def create_channel_names(mean_or_median_labels, stdev_labels):

    mean_or_median_labels = [f'm_{mean_or_median_label}'\
                for mean_or_median_label in mean_or_median_labels]

    stdev_labels = [f'std_{stdev_label}'\
                    for stdev_label in stdev_labels]

    labels = np.concatenate((mean_or_median_labels,
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

            output_category = rng.choice(choices, p=p_train)

            if output_category == 'train':
                output_subsets['unlabeled train'][id_].append(element)

                if rng.choice(2, p=[0.1, 0.9]):
                    output_subsets['labeled train'][id_].append(element)
                else:
                    # Add some data from the train set to the val
                    # sets to ensure that they include LAMMPS data
                    output_subsets['unlabeled val'][id_].append(element)
                    output_subsets['labeled val'][id_].append(element)

            elif output_category == 'test':
                output_subsets['test'][id_].append(element)

            else:
                output_subsets['unlabeled val'][id_].append(element)

                # We also add unlabeled val subsets to the
                # test set as it is otherwise heavily skewed
                # towards anomalous data

                output_subsets['test'][id_].append(element)

                if rng.choice(2, p=[0.1, 0.9]):
                    output_subsets['labeled val'][id_].append(element)


    for id_, data in test_subsets_in.items():
        for element in data:

            output_category = rng.choice(choices, p=p_test)

            if output_category == 'train':
                output_subsets['labeled train'][id_].append(element)

            elif output_category == 'test':
                output_subsets['test'][id_].append(element)

            else:
                output_subsets['labeled val'][id_].append(element)

    # values_unique_all = defaultdict(list)

    for dataset_type, dataset in output_subsets.items():

        print(f'{dataset_type}:')

        subset_list = []

        for id_, subsets in dataset.items():
            for subset in subsets:
                subset_list.append(subset)

        dataset_reshaped = pd.concat(subset_list, axis=1)

        dataset_reshaped.sort_index(inplace=True)

        dataset_reshaped.index =\
            remove_timestamp_jumps(dataset_reshaped.index)

        print(f'Length: {len(dataset_reshaped)}')

        nan_amount = np.mean(np.sum(pd.isna(dataset_reshaped.to_numpy()), 1)/\
                                                        dataset_reshaped.shape[1])

        print(f'Mean sparsity reshaped {dataset_type} set: {100*nan_amount:.3f} %')

        label_columns =\
            [col for col in dataset_reshaped.columns if 'label' in col]

        dataset_reshaped[label_columns] =\
            dataset_reshaped[label_columns].fillna(0).astype(np.uint8)

        labels_individual = dataset_reshaped[label_columns]

        labels_individual.columns =\
            rename_label_columns(labels_individual.columns)

        dataset_reshaped['label'] =\
            dataset_reshaped[label_columns]\
                .agg(lambda row: reduce(lambda x, y: x|y, row.tolist()), axis=1)
        
        dataset_reshaped.drop(label_columns, axis=1, inplace=True)

        # Print anomaly ratio

        anomaly_count =\
            np.count_nonzero(dataset_reshaped['label'].to_numpy().flatten()>=1)

        print(f'Anomalous data ratio: {100*anomaly_count/len(dataset_reshaped):.3f} %')

        # Sort columns by application and metric

        dataset_reshaped_ordered =\
            dataset_reshaped.loc[:, ~(dataset_reshaped.columns == 'label')].sort_index(axis=1)

        # for app in ['exa', 'lammps', 'sw4', 'sw4lite']:

        #     values_unique = []

        #     cols = dataset_reshaped_ordered.columns.str.startswith(f'{app}_')
            
        #     app_data = dataset_reshaped_ordered.loc[:, cols]

        #     for col in range(app_data.shape[-1]):
        #         label_local =\
        #             '_'.join(str(app_data.iloc[:, col].name).split('_')[:-1])

        #         values_unique_all[label_local].append(app_data.iloc[:, col].unique())

        # continue

        # Convert columns with less than 16 unique values
        # to ordinal representation

        ordinal_locs = dataset_reshaped_ordered\
                        .columns.str.startswith(ordinal_cols)

        # print(dataset_reshaped_ordered.columns[ordinal_locs])

        # exit()

        ordinal_data_df =\
            dataset_reshaped_ordered.loc[:, ordinal_locs]

        encoder = OrdinalEncoder()

        ordinal_data_np = encoder.fit_transform(ordinal_data_df)

        # ordinal_data_df = pd.DataFrame(ordinal_data_np,
        #                                 index=ordinal_data_df.index,
        #                                 columns=ordinal_data_df.columns)

        # for col in range(ordinal_data_df.shape[-1]):
        #     unique_vals = np.unique(ordinal_data_df.iloc[:, col])
        #     print(f'{ordinal_data_df.columns[col]}: {len(unique_vals)}')

        dataset_reshaped_ordered.loc[:, ordinal_locs] =\
                                            ordinal_data_np

        # print(ordinal_data)
        # print(encoder.categories_)

        # We also need the test set in unreduced form

        if dataset_type == 'test':

            # Add the labels back

            dataset_reshaped_ordered['label'] =\
                            dataset_reshaped['label']

            dataset_reshaped_ordered[label_columns] =\
                                        labels_individual

            dataset_label =\
                f"unreduced_eclipse_{dataset_type.replace(' ', '_')}_set"

            # dataset_reshaped_ordered.to_hdf(
            #             f'{args.dataset_dir}/{dataset_label}.h5',
            #             key=dataset_label, mode='w')
            
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

        # Plot meminfo data as a representation for the dataset

        apps = ['exa', 'lammps', 'sw4', 'sw4lite']

        plt.rcParams['figure.constrained_layout.use'] = True

        fig, axes = plt.subplots(4, 1, figsize=(10, 24), dpi=300)

        fig.suptitle(f'Eclipse {dataset_type.title()} '\
                                        'Set: MemInfo Data')

        meminfo_data =\
            dataset_reshaped_ordered.loc[:, dataset_reshaped_ordered.columns.str.contains('meminfo')]

        for ax, app in zip(axes, apps):

            ax.set_title(app.upper())
            
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('Data')

            ax.grid()

            meminfo_cols =  meminfo_data.columns.str.startswith(f'{app}_')

            data = meminfo_data.loc[:, meminfo_cols]

            if data.shape[-1]:

                data = data.fillna(0)

                data = MinMaxScaler().fit_transform(data.to_numpy())
                
                index = dataset_reshaped_ordered.index.values

                ax.plot(dataset_reshaped_ordered.index.values, data)

                anomaly_starts, anomaly_ends =\
                    get_contiguous_runs(dataset_reshaped['label'].to_numpy().flatten()>=1)
                    
                for start, end in zip(anomaly_starts, anomaly_ends):

                    start = max(0, start)
                    end = min(end, (len(index) - 1))

                    ax.axvspan(index[start], index[end], color='red', alpha=0.5)
            
        plt.savefig(f"plots/eclipse_{dataset_type.replace(' ', '_')}"\
                                                    '_set_meminfo.png')

        # Reduce dataset

        reduction_map =\
            [val.rsplit('_', 1)[0] for val in\
                dataset_reshaped_ordered.columns.values]

        data_mean_all = []
        data_median_all = []

        columns_reduced = None
        keys_last = None

        for count, row_x_data in enumerate(tqdm(dataset_reshaped_ordered.to_numpy(),
                                                desc=f'Generating {dataset_type} set')):

            buckets = defaultdict(list)

            for index, datapoint in enumerate(row_x_data):
                buckets[reduction_map[index]].append(datapoint)

            data_mean = {}
            data_median = {}
            data_std = {}

            for col_reduced, bucket in buckets.items():
                data_mean[col_reduced] = np.nanmedian(bucket)
                data_median[col_reduced] = np.nanmedian(bucket)
                data_std[col_reduced] = np.nanstd(bucket)

            data_mean = dict(sorted(data_mean.items()))
            data_median = dict(sorted(data_median.items()))
            data_std = dict(sorted(data_std.items()))

            if keys_last != None:
                assert data_median.keys() == keys_last,\
                                'Bucket keys changed between slices'

                assert data_median.keys() == data_std.keys(),\
                                                'Bucket keys not identical'

            keys_last = data_median.keys()

            if type(columns_reduced) == type(None):
                columns_reduced = create_channel_names(data_median.keys(),
                                                                data_std.keys())

            data_mean_np = np.concatenate((np.array(list(data_mean.values())),
                                                    np.array(list(data_std.values()))))

            data_median_np = np.concatenate((np.array(list(data_median.values())),
                                                    np.array(list(data_std.values()))))

            data_mean_all.append(data_mean_np)
            data_median_all.append(data_median_np)

        data_mean_all_np = np.stack(data_mean_all)
        data_median_all_np = np.stack(data_median_all)

        data_mean_all_np = np.nan_to_num(data_mean_all_np, nan=-1)
        data_median_all_np = np.nan_to_num(data_median_all_np, nan=-1)

        nan_amount_train_unlabeled =\
                100*pd.isna(data_median_all_np.flatten()).sum()/\
                                                data_median_all_np.size

        print(f'NaN amount: {nan_amount_train_unlabeled:.3f} %')
        
        data_mean_all_df = pd.DataFrame(data_mean_all_np,
                                            dataset_reshaped_ordered.index,
                                            columns_reduced)

        data_mean_all_df['label'] = dataset_reshaped['label']
        
        data_mean_all_df[label_columns] = labels_individual

        data_median_all_df = pd.DataFrame(data_median_all_np,
                                            dataset_reshaped_ordered.index,
                                            columns_reduced)

        data_median_all_df['label'] = dataset_reshaped['label']

        data_median_all_df[label_columns] = labels_individual

        dataset_label = f"reduced_eclipse_{dataset_type.replace(' ', '_')}_set"

        # data_mean_all_df.to_hdf(f'{args.dataset_dir}/{dataset_label}_mean.h5',
        #                                             key=dataset_label, mode='w')
        
        # data_median_all_df.to_hdf(f'{args.dataset_dir}/{dataset_label}_median.h5',
        #                                                 key=dataset_label, mode='w')

        if args.generate_videos:

            four_cc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

            writer = cv.VideoWriter(f'{args.video_output_dir}/{dataset_label}.mp4',
                                            four_cc, 60, (image_width, image_height))

            for count in tqdm(range(len(data_mean_all_np)),
                                desc='Generating unlabeled train set animation'):

                lower_bound = max(count - plot_window_size, 0)
                upper_bound_axis = max(count, plot_window_size) + 10

                fig, (ax_mean, ax_median) =\
                        plt.subplots(2, 1, figsize=(8, 9), dpi=240)

                max_val_slice_mean =\
                    np.max(data_mean_all_np[lower_bound:count, :])\
                            if len(data_mean_all_np[lower_bound:count, :])\
                        else 10
                
                max_val_slice_median =\
                    np.max(data_median_all_np[lower_bound:count, :])\
                            if len(data_median_all_np[lower_bound:count, :])\
                        else 10

                max_val_slice_mean = min(max_val_slice_mean, 100000)
                max_val_slice_median = min(max_val_slice_mean, 1000000)

                ax_mean.set_xlim(lower_bound, upper_bound_axis)
                ax_mean.set_ylim(-2, max_val_slice_mean + 10)
                ax_median.set_xlim(lower_bound, upper_bound_axis)
                ax_median.set_ylim(-2, max_val_slice_mean + 10)

                ax_mean.grid(True)
                ax_median.grid(True)

                ax_mean.set_title('Mean Data')
                ax_mean.set_xlabel('Timestep')
                ax_mean.set_ylabel('Mean Data')

                ax_mean.set_title('Median Data')
                ax_mean.set_xlabel('Timestep')
                ax_mean.set_ylabel('Median Data')

                ax_mean.plot(np.arange(lower_bound, count),
                                data_mean_all_np[lower_bound:count, :])
                
                ax_median.plot(np.arange(lower_bound, count),
                                data_median_all_np[lower_bound:count, :])

                frame = fig_to_numpy_array(fig)

                writer.write(frame)

                plt.close()

            writer.release()

    # columns = []

    # for name, values_unique_per_channel in values_unique_all.items():
    #     values = np.concatenate(values_unique_per_channel)

    #     values_unique = np.unique(values)

    #     if len(values_unique) < 16:
    #         columns.append(name)

    # print(columns)

    # columns_all = ['_'.join(val.split('_')[1:]) for val in columns]

    # from collections import Counter

    # counts = Counter(columns_all)

    # print(counts)
