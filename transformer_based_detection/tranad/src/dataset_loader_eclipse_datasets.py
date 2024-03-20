import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .data_augmentation import EclipseDataTimeseriesAugmentor

import warnings
warnings.filterwarnings('ignore')

dataset_path_local = '../../datasets/eclipse/'


class EclipseDataset(Dataset):
    def __init__(self,
                    mode,
                    inverse,
                    scaling_type,
                    scaling_source,
                    applied_augmentations=[],
                    augmented_dataset_size_relative=1,
                    augmented_data_ratio=0):

        self.mode = mode
        self.inverse = inverse

        if scaling_type == 'standard':
            self.scaler = StandardScaler()
        elif scaling_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise NotImplementedError(
                        'Invalid dataset scaler')

        if mode == 'train' or mode == 'unlabeled_train':

            data_unlabeled_train =\
                    pd.read_hdf(dataset_path_local +\
                                    'reduced_eclipse_'
                                    'unlabeled_train_set_median.h5')

            data_train_set_x_pd =\
                data_unlabeled_train.loc[:, ~(data_unlabeled_train.columns.str.contains('label'))]

            data_train_set_x_pd.index *= 1e9

            data_train_set_x_pd.index =\
                pd.DatetimeIndex(data_train_set_x_pd.index)

            if len(applied_augmentations):

                timeseries_augmentor =\
                    EclipseDataTimeseriesAugmentor(applied_augmentations)

                data_train_set_x_pd =\
                    timeseries_augmentor.fit_transform(data_train_set_x_pd)

            data_train_set_x_np = data_train_set_x_pd.to_numpy()

            self.scaler.fit(data_train_set_x_np)

            data_x_pd = data_train_set_x_pd
            data_x_np = data_train_set_x_np

        if mode == 'labeled_train':

            data_labeled_train = pd.read_hdf(dataset_path_local +\
                                                'reduced_eclipse_'
                                                'labeled_train_set_median.h5')

            data_labeled_train_set_x_pd = \
                data_labeled_train.loc[:, ~(data_labeled_train.columns.str.contains('label'))]

            labels_pd =\
                data_labeled_train.loc[:, (data_labeled_train.columns.str.contains('label'))]

            data_labeled_train_set_x_pd.index =\
                pd.DatetimeIndex(data_labeled_train_set_x_pd.index)

            labels_pd.index = pd.DatetimeIndex(labels_pd.index)

            if len(applied_augmentations):
                timeseries_augmentor =\
                    EclipseDataTimeseriesAugmentor(applied_augmentations)

                target_size = int(len(data_labeled_train_set_x_pd)*\
                                        augmented_dataset_size_relative)

                data_labeled_train_set_x_pd,\
                                    labels_pd =\
                                        timeseries_augmentor.fit_transform_labeled(
                                                                data_labeled_train_set_x_pd,
                                                                labels_pd)

                data_labeled_train_set_x_pd =\
                    data_labeled_train_set_x_pd.iloc[:target_size, :]

                labels_pd =\
                    labels_pd.iloc[:target_size, :]

            data_labeled_train_set_x_np = data_labeled_train_set_x_pd.to_numpy()

            self.labels = labels_pd.to_numpy().astype(np.int8).flatten()

            if scaling_source == 'train_set_fit':
                data_unlabeled_train =\
                        pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'unlabeled_train_set_median.h5')

                data_unlabeled_train_set_x_pd =\
                    data_unlabeled_train.loc[:, ~(data_unlabeled_train.columns.str.contains('label'))]

                data_unlabeled_train_set_x_np =\
                            data_unlabeled_train_set_x_pd.to_numpy()

                self.scaler.fit(data_unlabeled_train_set_x_np)

            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data_labeled_train_set_x_np)
            else:
                raise RuntimeError(
                        'Invalid dataset scaling source')

            data_x_pd = data_labeled_train_set_x_pd
            data_x_np = data_labeled_train_set_x_np
            
        elif mode == 'test':

            data_test = pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'test_set_median.h5')

            data_x_pd =\
                data_test.loc[:, ~(data_test.columns.str.contains('label'))]

            labels_pd =\
                data_test.loc[:, data_test.columns == 'label']
            
            print(labels_pd.columns)
            exit()

            self.labels = np.any(labels_pd.to_numpy()>=1, axis=1).astype(np.int8).flatten()

            print(self.labels)
            
            data_x_pd.index = pd.DatetimeIndex(data_x_pd.index)

            data_x_np = data_x_pd.to_numpy()

            if scaling_source == 'train_set_fit':
                data_unlabeled_train =\
                        pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'unlabeled_train_set_median.h5')

                data_unlabeled_train_set_x_pd =\
                    data_unlabeled_train.loc[:, ~(data_unlabeled_train.columns.str.contains('label'))]

                data_train_set_x_np =\
                    data_unlabeled_train_set_x_pd.to_numpy()

                self.scaler.fit(data_train_set_x_np)

            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data_x_np)
            else:
                raise RuntimeError(
                        'Invalid dataset scaling source')

        elif mode == 'val':

            data_val = pd.read_hdf(dataset_path_local +\
                                            'reduced_eclipse_'
                                            'unlabeled_val_set_median.h5')

            data_x_pd =\
                data_val.loc[:, ~(data_val.columns.str.contains('label'))]
            
            data_x_pd.index = pd.DatetimeIndex(data_x_pd.index)
            data_x_np = data_x_pd.to_numpy()

            if scaling_source == 'train_set_fit':
                data_unlabeled_train =\
                        pd.read_hdf(dataset_path_local +\
                                        'reduced_eclipse_'
                                        'unlabeled_train_set_median.h5')

                data_unlabeled_train_set_x_pd =\
                    data_unlabeled_train.loc[:, ~(data_unlabeled_train.columns.str.contains('label'))]

                data_train_set_x_np =\
                    data_unlabeled_train_set_x_pd.to_numpy()

                self.scaler.fit(data_train_set_x_np)

            elif scaling_source == 'individual_set_fit':
                self.scaler.fit(data_x_np)
            else:
                raise RuntimeError(
                        'Invalid scaling source')

        self.data_x = self.scaler.transform(data_x_np)

        self.shape = self.data_x.shape

        if np.any(np.isnan(self.data_x)):
            raise RuntimeError(f'NaN in transformed {mode} data')

        if self.inverse:
            self.data_y = data_x_np
        else:
            self.data_y = self.data_x


    def get_data(self):
        return self.data_x


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


    def get_channels(self):
        return len(self.data_x[0, :])


    def get_labels(self):
        return self.labels


    def pickle_scaler(self,
                        filename: str) -> None:
        pickle.dump(self.scaler, open(filename, 'wb'))