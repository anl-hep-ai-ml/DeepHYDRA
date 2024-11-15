#!/usr/bin/env python3

from abc import ABC, abstractmethod
import logging

import numpy as np
#from pandas import DataFrame
import pandas as pd

from ..utils.channellabels import subgroup_labels_expected_hlt_dcm_2018,\
                                    subgroup_labels_expected_hlt_dcm_2023,\
                                    subgroup_labels_expected_eclipse
                                    

class BaseReducer(ABC):

    def __init__(self, configuration_version: str) -> None:
        self._configuration_version = configuration_version

        if self._configuration_version == '2018':
            self._subgroup_numbers_expected = subgroup_labels_expected_hlt_dcm_2018
        elif self._configuration_version == '2023':
            self._subgroup_numbers_expected = subgroup_labels_expected_hlt_dcm_2023
        elif self._configuration_version == 'ECLIPSE':
            self._subgroup_numbers_expected = subgroup_labels_expected_eclipse
        else:
            raise ValueError('Configuration version '
                                f'{self._configuration_version} '
                                'is unknown')

        self._logger = logging.getLogger(__name__)
        self._missing_subgroups_feedback_given = False

    @abstractmethod
    def reduce_pandas(self, input_slice: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def reduce_numpy(self,
                        input_slice: np.array,
                        tpu_labels: list,
                        timestamps: list) -> pd.DataFrame:
        pass


    def _adjust_reduced_data(self,
                                labels_reduced: np.array,
                                data_reduced: np.array) -> np.array:

        subgroup_count_expected = len(self._subgroup_numbers_expected)
        subgroup_count_observed = len(labels_reduced)//2

        if subgroup_count_observed < subgroup_count_expected:
            subgroup_numbers_observed =\
                    [int(label.removeprefix('m_'))\
                        for label in labels_reduced[:subgroup_count_observed]]

            missing_subgroups = np.setdiff1d(self._subgroup_numbers_expected,
                                                subgroup_numbers_observed)

            indices_missing =\
                    np.nonzero(np.isin(self._subgroup_numbers_expected,
                                                    missing_subgroups))[0]

            if not self._missing_subgroups_feedback_given:
                
                missing_subgroups_string = ''

                for subgroup in missing_subgroups:
                    missing_subgroups_string += f'{subgroup}, '

                self._logger.warning(f'Rack(s) {missing_subgroups_string} are '
                                        'missing. 2nd stage detection '
                                        'performance might be affected.')

                # missing_subgroup_indices_string =\
                #             ', '.join(str(indices_missing))\
                #                 if len(indices_missing) > 1\
                #                 else str(indices_missing[0])

                # self._logger.debug('Indices missing subgroups: '
                #                     f'{missing_subgroup_indices_string}')

                self._missing_subgroups_feedback_given = True

            data_reduced = np.insert(data_reduced,
                                        indices_missing,
                                        0, axis=1)

            data_reduced = np.insert(data_reduced,
                                        indices_missing +\
                                            subgroup_count_expected,
                                        0, axis=1)

            missing_labels_median =\
                [f'm_{subgroup}' for subgroup in missing_subgroups]

            labels_reduced = np.insert(labels_reduced,
                                        indices_missing,
                                        missing_labels_median)

            missing_labels_std =\
                [f'std_{subgroup}' for subgroup in missing_subgroups]

            labels_reduced = np.insert(labels_reduced,
                                        indices_missing +\
                                            subgroup_count_expected,
                                        missing_labels_std)

        return labels_reduced, data_reduced



    # # VERSION WITH DATAFRAMES
    # def _adjust_reduced_data(self, labels_reduced: np.array, data_reduced: np.array) -> (np.array, np.array):
    #     # Convert the NumPy array to a DataFrame
    #     racks = self._subgroup_numbers_expected
    #     columns = list(labels_reduced)  # Assuming labels_reduced corresponds to data_reduced columns
    #     data_reduced_df = pd.DataFrame(data_reduced, columns=columns)
        
    #     # Perform the adjustment using the DataFrame
    #     adjusted_data_df = self.df_adjust_reduced_data(data_reduced_df)
        
    #     # Convert the adjusted DataFrame back to a NumPy array
    #     adjusted_labels_reduced = adjusted_data_df.columns.to_numpy()
    #     adjusted_data_reduced = adjusted_data_df.to_numpy()

    #     return adjusted_labels_reduced, adjusted_data_reduced

    # def df_adjust_reduced_data(self, data_reduced: pd.DataFrame) -> pd.DataFrame:
    #     racks = self._subgroup_numbers_expected
    #     all_columns = [f'm_{rack}' for rack in racks] + [f'std_{rack}' for rack in racks]
        
    #     # Reindex the DataFrame to include all expected columns, filling missing with NaN
    #     data_reduced = data_reduced.reindex(columns=all_columns)

    #     # Extract rack numbers from columns, drop NaNs before conversion
    #     extracted_racks = data_reduced.columns.str.extract(r'm_(\d+)')[0].dropna().astype(int)

    #     # Identify missing racks
    #     missing_racks = set(racks) - set(extracted_racks)
    #     if missing_racks:
    #         missing_racks_str = ', '.join(map(str, missing_racks))
    #         self._logger.warning(f'Rack(s) {missing_racks_str} are missing. 2nd stage detection performance might be affected.')

    #     # Fill any NaN values with zeros to avoid conversion issues
    #     data_reduced.fillna(0, inplace=True)


    #     return data_reduced



    # # ORIGINAL VERSION
    # def _adjust_reduced_data(self,
    #                             labels_reduced: np.array,
    #                             data_reduced: np.array) -> np.array:

    #     #print(f"labels_reduced shape: {labels_reduced.shape}")
    #     #print(f"data_reduced shape: {data_reduced.shape}")
        

    #     subgroup_count_expected = len(self._subgroup_numbers_expected)
    #     subgroup_count_observed = len(labels_reduced)//2

    #     #print(f"subgroup_count_expected {subgroup_count_expected}")
    #     #print(f"subgroup_count_observed {subgroup_count_observed}")


    #     if subgroup_count_observed < subgroup_count_expected:
    #         subgroup_numbers_observed =\
    #                 [int(label.removeprefix('m_'))\
    #                     for label in labels_reduced[:subgroup_count_observed]]

    #         missing_subgroups = np.setdiff1d(self._subgroup_numbers_expected,
    #                                             subgroup_numbers_observed)

    #         indices_missing =\
    #                 np.nonzero(np.isin(self._subgroup_numbers_expected,
    #                                                 missing_subgroups))[0]

    #         #print(f"indices_missing: {indices_missing}")
    #         #print(f"missing_subgroups {missing_subgroups}")
    #         #print(f"indices_missing {indices_missing}")
    #         if not self._missing_subgroups_feedback_given:
                
    #             #missing_subgroups_string = ''
    #             #for subgroup in missing_subgroups:
    #             #    missing_subgroups_string += f'{subgroup}, '

    #             missing_subgroups_string = ', '.join(map(str, missing_subgroups))

    #             self._logger.warning(f'Rack(s) {missing_subgroups_string} are '
    #                                     'missing. 2nd stage detection '
    #                                     'performance might be affected.')

    #             # missing_subgroup_indices_string =\
    #             #             ', '.join(str(indices_missing))\
    #             #                 if len(indices_missing) > 1\
    #             #                 else str(indices_missing[0])

    #             # self._logger.debug('Indices missing subgroups: '
    #             #                     f'{missing_subgroup_indices_string}')

    #             self._missing_subgroups_feedback_given = True

    #         # # Pre-expand data_reduced to have the correct number of columns
    #         # num_new_columns = subgroup_count_expected * 2 - data_reduced.shape[1]
    #         # if num_new_columns > 0:
    #         #     data_reduced = np.hstack([data_reduced, np.zeros((data_reduced.shape[0], num_new_columns))])


    #         # num_new_labels = subgroup_count_expected * 2 - len(labels_reduced)
    #         # if num_new_labels > 0:
    #         #     labels_reduced = np.hstack([labels_reduced, [''] * num_new_labels])

    #         # Insert zeros for all missing subgroups at once
    #         for idx, insert_idx in enumerate(indices_missing):
    #             # Insert median columns
    #             data_reduced = np.insert(data_reduced, insert_idx + idx, 0, axis=1)
    #             labels_reduced = np.insert(labels_reduced, insert_idx + idx, f'm_{missing_subgroups[idx]}')

    #         # Adjust std insertions by offsetting them to account for the already inserted medians
    #         for idx, insert_idx in enumerate(indices_missing):
    #             adjusted_idx = insert_idx + subgroup_count_expected + idx
    #             data_reduced = np.insert(data_reduced, adjusted_idx, 0, axis=1)
    #             labels_reduced = np.insert(labels_reduced, adjusted_idx, f'std_{missing_subgroups[idx]}')

    #         # #This should work but it was throwing many index errors
    #         # data_reduced = np.insert(data_reduced,
    #         #                             indices_missing,
    #         #                             0, axis=1)

    #         # data_reduced = np.insert(data_reduced,
    #         #                             indices_missing +\
    #         #                                 subgroup_count_expected,
    #         #                             0, axis=1)

    #         # missing_labels_median =\
    #         #     [f'm_{subgroup}' for subgroup in missing_subgroups]

    #         # labels_reduced = np.insert(labels_reduced,
    #         #                             indices_missing,
    #         #                             missing_labels_median)

    #         # missing_labels_std =\
    #         #     [f'std_{subgroup}' for subgroup in missing_subgroups]

    #         # labels_reduced = np.insert(labels_reduced,
    #         #                             indices_missing +\
    #         #                                 subgroup_count_expected,
    #         #                             missing_labels_std)

    #     else:
    #         labels_reduced_adjusted = labels_reduced
    #         data_reduced_adjusted = data_reduced

    #     return labels_reduced_adjusted, data_reduced_adjusted        

    #     #return labels_reduced, data_reduced


    # def _adjust_reduced_data(self,
    #                              labels_reduced: np.array,
    #                              data_reduced: np.array) -> np.array:

    #     subgroup_count_expected = len(self._subgroup_numbers_expected)
    #     subgroup_count_observed = len(labels_reduced)//2

    #     if subgroup_count_observed < subgroup_count_expected:

    #         if not self._missing_subgroups_feedback_given:
    #             self._logger.warning(f'Some Racks are missing... 2nd stage detection performance might be affected.')
    #             self._missing_subgroups_feedback_given = True

    #         # Flatten the data array to handle as a 1D array
    #         data_reduced = data_reduced.flatten()

    #         # Extract the rack numbers from 'm_' and 'std_' labels. With only one should be enough. Improve!!!
    #         m_mask = np.char.startswith(labels_reduced, 'm_')
    #         std_mask = np.char.startswith(labels_reduced, 'std_')

    #         current_m_racks = np.array([int(label.split('_')[1]) for label in labels_reduced[m_mask]])
    #         current_std_racks = np.array([int(label.split('_')[1]) for label in labels_reduced[std_mask]])

    #         # Dictionary to hold the data for faster lookups
    #         m_data_dict = dict(zip(current_m_racks, data_reduced[m_mask]))
    #         std_data_dict = dict(zip(current_std_racks, data_reduced[std_mask]))

    #         # Preallocate new labels and data arrays
    #         m_labels = [f'm_{rack}' for rack in self._subgroup_numbers_expected]
    #         std_labels = [f'std_{rack}' for rack in self._subgroup_numbers_expected]

    #         m_data = np.array([m_data_dict.get(rack, 0) for rack in self._subgroup_numbers_expected])
    #         std_data = np.array([std_data_dict.get(rack, 0) for rack in self._subgroup_numbers_expected])

    #         labels_reduced_adjusted = np.array(m_labels + std_labels)
    #         data_reduced_adjusted = np.concatenate((m_data, std_data))
    #         # Reshape the data to be 2D (1, N)
    #         data_reduced_adjusted = data_reduced_adjusted.reshape(1, -1)

    #     else:
    #         labels_reduced_adjusted = labels_reduced
    #         data_reduced_adjusted = data_reduced

    #     return labels_reduced_adjusted, data_reduced_adjusted       
