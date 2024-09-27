import numpy as np
import pandas as pd
import torch
import pickle as pkl
import json
import logging

from .timefeatures import time_features
from sklearn.preprocessing import MinMaxScaler
from utils.exceptions import PredictionInputException
from utils.variables import nan_fill_value

class DataPreprocessor():
    def __init__(self,
                    parameter_dict: dict,
                    checkpoint_dir: str,
                    device: str = 'cpu',
                    data_clipping_val: float = 1e9) -> None:

        self.device = torch.device(device)
        self.seq_len = parameter_dict['seq_len']
        self.label_len = parameter_dict['label_len']
        self.pred_len = parameter_dict['pred_len']

        self.features = 'M'
        self.timeenc = parameter_dict['timeenc']
        self.freq = parameter_dict['freq']
        self.inverse = parameter_dict['inverse']
        
        # scaler_dir_and_filename =\
        #             checkpoint_dir +\
        #             '/scaler.pkl'

        scaler_dir_and_filename =\
                    checkpoint_dir +\
                    '/scaler_params.json'

        # This is a hack to avoid the issues between different versions of the scaler
        self.scaler = MinMaxScaler()
        # Load the scaler using a context manager
        with open(scaler_dir_and_filename, 'rb') as file:
            #old_scaler = pkl.load(file)
            scaler_params = json.load(file)
            
        self.scaler.min_ = np.array(scaler_params['min_'])
        self.scaler.scale_ = np.array(scaler_params['scale_'])
        self.scaler.data_min_ = np.array(scaler_params['data_min_'])
        self.scaler.data_max_ = np.array(scaler_params['data_max_'])
        self.scaler.feature_range = tuple(scaler_params['feature_range'])
        ###self.scaler = pkl.load(open(
        ###                scaler_dir_and_filename, 'rb'))


        self._data_clipping_val =\
                    data_clipping_val

        self._logger = logging.getLogger(__name__)
        self._missing_tpus_feedback_given = False



    # def process(self, data: pd.DataFrame):

    #     # Data is a df with all the medians and std deviation for all racks.
    #     timestamps = pd.DataFrame(data.index,
    #                                 columns=['date'])

    #     data_x = data.to_numpy()
    #     #if np.any(data_x == nan_fill_value):
    #     if np.any(np.isclose(data_x, nan_fill_value)):
    #         _, data_x_nan_indices =\
    #             np.nonzero(np.isclose(data_x, nan_fill_value))
    #             #np.nonzero(data_x == nan_fill_value)

    #         missing_racks =\
    #             [rack.removeprefix('m_')\
    #                 for rack in data.columns[data_x_nan_indices]]

    #         missing_racks = list(dict.fromkeys(missing_racks))

    #         if len(missing_racks) == 1:
    #             missing_racks_formatted = missing_racks[0]
    #         else:
    #             missing_racks_formatted =\
    #                     ', '.join(missing_racks)

    #         if not self._missing_tpus_feedback_given:
    #             warning_string =\
    #                 'Half or more of the TPUs in rack(s) '\
    #                 f'{missing_racks_formatted} '\
    #                 'are inactive. Second stage '\
    #                 'detection might be affected'

    #             self._logger.warning(warning_string)
    #             self._missing_tpus_feedback_given = True


    #     # Clip large positive/negative values
    #     # that sometimes occur when a sufficiently
    #     # large number of values in an unreduced
    #     # slice are large positive/negative values
    #     # sometimes returned by PBEAST

    #     np.clip(data_x,
    #                 -self._data_clipping_val,
    #                 self._data_clipping_val,
    #                 out=data_x)


    #     data_x_scaled = self.scaler.transform(data_x)

    #     # for element in data_x:
    #     #     print(element)

    #     # for element in data_x_scaled:
    #     #     print(element)

    #     if self.inverse:
    #         data_y = data_x
    #     else:
    #         data_y = data_x_scaled

    #     data_x = data_x_scaled

    #     # print(f'Scaled data mean {data_x.mean()}')

    #     data_stamp = time_features(timestamps,
    #                                 timeenc=self.timeenc,
    #                                 freq=self.freq)

    #     s_begin = 0
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len 
    #     r_end = r_begin + self.label_len + self.pred_len

    #     seq_x = data_x[s_begin:s_end]

    #     if self.inverse:
    #         seq_y = np.concatenate([data_x[r_begin:r_begin + self.label_len],
    #                                     data_y[r_begin + self.label_len:r_end]], 0)
    #     else:
    #         seq_y = data_y[r_begin:r_end]

    #     seq_x_mark = data_stamp[s_begin:s_end]
    #     seq_y_mark = data_stamp[r_begin:r_end]

    #     seq_x = torch.from_numpy(seq_x)
    #     seq_y = torch.from_numpy(seq_y)
    #     seq_x_mark = torch.from_numpy(seq_x_mark)
    #     seq_y_mark = torch.from_numpy(seq_y_mark)

    #     if seq_x.dim() == 2:
    #         seq_x = torch.unsqueeze(seq_x, 0)
    #         seq_y = torch.unsqueeze(seq_y, 0)
    #         seq_x_mark = torch.unsqueeze(seq_x_mark, 0)
    #         seq_y_mark = torch.unsqueeze(seq_y_mark, 0)

    #     return seq_x, seq_y, seq_x_mark, seq_y_mark



    def process(self, data: pd.DataFrame):
        # Convert data to NumPy array with specified dtype
        data_x = data.values.astype(np.float32)

        # Detect NaN values
        nan_mask = np.isclose(data_x, nan_fill_value)

        if np.any(nan_mask):
            # Find the indices of NaN values, get unique column indices where NaNs are present and the rack names.
            # Then we remove the prefix and duplicates keeping the order
            data_x_nan_indices = np.where(nan_mask)
            nan_column_indices = np.unique(data_x_nan_indices[1])
            nan_rack_names = data.columns[nan_column_indices]
            missing_racks = [rack.removeprefix('m_') for rack in nan_rack_names]
            missing_racks = list(dict.fromkeys(missing_racks))

            # Format the missing racks for logging
            if len(missing_racks) == 1:
                missing_racks_formatted = missing_racks[0]
            else:
                missing_racks_formatted = ', '.join(missing_racks)

            if not self._missing_tpus_feedback_given:
                warning_string = (
                    f'Half or more of the TPUs in rack(s) {missing_racks_formatted} are inactive. '
                    'Second stage detection might be affected'
                )
                self._logger.warning(warning_string)
                self._missing_tpus_feedback_given = True

            #data_x[nan_mask] = self.nan_replacement_value

        # Clip extreme values in-place
        np.clip(data_x, -self._data_clipping_val, self._data_clipping_val, out=data_x)

        # Scale data
        data_x_scaled = self.scaler.transform(data_x)

        data_y = data_x if self.inverse else data_x_scaled
        data_x = data_x_scaled

        # Generate time features
        timestamps = pd.DataFrame({'date': data.index})
        data_stamp = time_features(timestamps, timeenc=self.timeenc, freq=self.freq)

        # Prepare sequences
        s_begin = 0
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([data_x[r_begin:r_begin + self.label_len],
                                    data_y[r_begin + self.label_len:r_end]], axis=0)
        else:
            seq_y = data_y[r_begin:r_end]

        seq_x_mark = data_stamp[s_begin:s_end]
        seq_y_mark = data_stamp[r_begin:r_end]

        # Convert to tensors and add batch dimension
        seq_x = torch.from_numpy(seq_x).unsqueeze(0).float().to(self.device)
        seq_y = torch.from_numpy(seq_y).unsqueeze(0).float().to(self.device)
        seq_x_mark = torch.from_numpy(seq_x_mark).unsqueeze(0).float().to(self.device)
        seq_y_mark = torch.from_numpy(seq_y_mark).unsqueeze(0).float().to(self.device)

        return seq_x, seq_y, seq_x_mark, seq_y_mark




    def register_data_provider_fn(self,
                                    data_provider_fn) -> None:
        self.data_provider_fn =\
                        data_provider_fn


    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


    def get_sequence_length(self):
        return self.seq_len